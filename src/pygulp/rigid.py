from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from pygulp.io.read_gulp import read_results
from pygulp.molecule.connections import (
    infer_molecule_tags_natural_cutoffs,
    infer_natural_cutoff_connections,
)


DEFAULT_KEYWORDS = "gradient conp gfnff gwolf c6 noauto"
DEFAULT_OPTIONS = "gfnff_scale 0.8 1.343 0.727 1.0 2.859\nmaths mrrr"
GPA_A3_TO_EV = 0.006241509074


def _exp_so3(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1.0e-14:
        return np.eye(3)
    axis = vector / theta
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _exp_symmetric(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.exp(values)) @ vectors.T


def _nullspace(matrix: np.ndarray, tolerance: float = 1.0e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1])
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance * max(1.0, singular_values[0])))
    return vh[rank:].T


def _unwrap_groups(atoms: Atoms, groups: list[np.ndarray]) -> None:
    cell = np.asarray(atoms.cell, dtype=float)
    scaled = atoms.get_scaled_positions(wrap=False)
    positions = atoms.get_positions().copy()
    for group in groups:
        if len(group) < 2:
            continue
        reference = int(group[0])
        for index in group[1:]:
            index = int(index)
            delta = scaled[index] - scaled[reference]
            delta -= np.round(delta)
            positions[index] = positions[reference] + delta @ cell
    atoms.set_positions(positions)


def _select_spacegroup(calc_dir: Path, options: dict[str, object], default_number: int) -> int:
    value = options.get("spacegroup", "auto")
    if isinstance(value, dict):
        value = value.get("number", value.get("spacegroup", "auto"))
    if str(value).strip().lower() in {"", "auto", "none"}:
        path = calc_dir.parent / "symmetry.json"
        if path.exists():
            return int(json.loads(path.read_text()).get("gulp_spacegroup_number", default_number))
        return int(default_number)
    number = int(value)
    if not 1 <= number <= 230:
        raise ValueError("symmetry.spacegroup must be between 1 and 230")
    return number


def _symmetry_operations(full: Atoms, number: int, tolerance: float):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.groups import SpaceGroup

    if number == 1:
        return [(np.eye(3), np.zeros(3))]
    structure = AseAtomsAdaptor.get_structure(full)
    analyzer = SpacegroupAnalyzer(structure, symprec=max(tolerance, 1.0e-3), angle_tolerance=5.0)
    if int(analyzer.get_space_group_number()) == number:
        operations = analyzer.get_symmetry_operations(cartesian=False)
    else:
        operations = SpaceGroup.from_int_number(number).symmetry_ops
    return [
        (np.asarray(op.rotation_matrix, dtype=float), np.asarray(op.translation_vector, dtype=float))
        for op in operations
    ]


def _apply_asu_parameters(
    reference: Atoms,
    groups: list[np.ndarray],
    translations: np.ndarray,
    rotations: np.ndarray,
    cell: np.ndarray,
    reference_centers: np.ndarray,
    center_fractions: np.ndarray,
) -> Atoms:
    atoms = reference.copy()
    positions = reference.get_positions().copy()
    for group_index, group in enumerate(groups):
        indices = np.asarray(group, dtype=int)
        center = center_fractions[group_index] @ cell + translations[group_index]
        local = reference.get_positions()[indices] - reference_centers[group_index]
        positions[indices] = local @ rotations[group_index].T + center
    atoms.set_cell(cell)
    atoms.set_positions(positions)
    return atoms


def _expand_asu(
    asu: Atoms,
    groups: list[np.ndarray],
    operations: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float,
    allow_special_positions: bool,
) -> tuple[Atoms, list[tuple[int, int, np.ndarray]], bool]:
    cell = np.asarray(asu.cell, dtype=float)
    scaled = asu.get_scaled_positions(wrap=True)
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    charges: list[float] = []
    tags: list[int] = []
    records: list[tuple[int, int, np.ndarray]] = []
    special = False
    charges_in = np.asarray(asu.get_initial_charges(), dtype=float)
    group_of_atom = np.full(len(asu), -1, dtype=int)
    for group_index, group in enumerate(groups):
        group_of_atom[group] = group_index

    for operation_index, (rotation, translation) in enumerate(operations):
        transformed = scaled @ rotation.T + translation
        for atom_index, (symbol, position) in enumerate(zip(asu.get_chemical_symbols(), transformed)):
            position = np.mod(position, 1.0)
            duplicate = False
            for old_symbol, old_position in zip(symbols, positions):
                if old_symbol != symbol:
                    continue
                delta = position - old_position
                delta -= np.round(delta)
                if float(np.linalg.norm(delta @ cell)) <= tolerance:
                    duplicate = True
                    break
            if duplicate:
                special = True
                continue
            symbols.append(symbol)
            positions.append(position)
            tags.append(int(group_of_atom[atom_index]))
            charges.append(float(charges_in[atom_index]) if len(charges_in) == len(asu) else 0.0)
            records.append((atom_index, operation_index, rotation))

    if special and not allow_special_positions:
        raise ValueError(
            "The symmetry-constrained rigid stage found special positions. "
            "Set symmetry.allow_special_positions=true only after validating the molecular mapping."
        )
    expanded = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    expanded.set_initial_charges(charges)
    expanded.set_tags(tags)
    return expanded, records, special


def _cell_strain_basis(cell: np.ndarray, operations, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "isotropic":
        return np.eye(3, dtype=float)[None, :, :] / np.sqrt(3.0)
    if mode not in {"symmetry", "anisotropic"}:
        raise ValueError("rigid.cell_mode must be symmetry or isotropic")

    inv_cell = np.linalg.inv(cell)
    constraints: list[np.ndarray] = []
    for rotation, _ in operations:
        q = inv_cell @ rotation.T @ cell
        for row in range(3):
            for column in range(3):
                elementary = np.zeros((3, 3))
                elementary[row, column] = 1.0
                constraints.append((q @ elementary - elementary @ q).reshape(-1))
    # Strain is symmetric. Antisymmetric cell changes are rotations of the basis.
    for row in range(3):
        for column in range(row + 1, 3):
            elementary = np.zeros((3, 3))
            elementary[row, column] = 1.0
            elementary[column, row] = -1.0
            constraints.append(elementary.reshape(-1))
    null = _nullspace(np.asarray(constraints))
    basis = []
    for column in range(null.shape[1]):
        matrix = null[:, column].reshape(3, 3)
        matrix = 0.5 * (matrix + matrix.T)
        norm = float(np.linalg.norm(matrix))
        if norm > 1.0e-8:
            basis.append(matrix / norm)
    if not basis:
        raise ValueError("Could not construct a symmetry-compatible cell strain basis")
    return np.asarray(basis)


def _render_input(atoms: Atoms, keywords: str, options: str) -> str:
    lines = [keywords.rstrip(), "title", "pyGulp symmetry-constrained rigid GFNFF", "end", "", "cell"]
    lines.append("{:9.6f} {:9.6f} {:9.6f} {:8.5f} {:8.5f} {:8.5f}".format(*atoms.cell.cellpar()))
    lines.append("frac")
    charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    if len(charges) != len(atoms):
        charges = np.zeros(len(atoms), dtype=float)
    for symbol, position, charge in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=True), charges):
        lines.append(
            f" {symbol:<2} {position[0]:10.7f}  {position[1]:10.7f}  "
            f"{position[2]:10.7f}  {charge:10.5f}"
        )
    if options.strip():
        lines.extend(["", options.rstrip()])
    return "\n".join(lines).rstrip() + "\n"


def _run_gulp(command_template: str, prefix: str, calc_dir: Path) -> int:
    return int(subprocess.run(command_template.replace("PREFIX", prefix), cwd=calc_dir, shell=True, check=False).returncode)


def _prepare_keywords(stage_keywords: str, options: dict[str, object]) -> tuple[str, float]:
    forbidden = {"opti", "optimise", "optimize", "rigid", "molecule", "spacegroup", "conv"}
    lines = []
    for line in stage_keywords.splitlines():
        kept = [word for word in line.split() if word.lower() not in forbidden]
        if kept:
            lines.append(" ".join(kept))
    keywords = "\n".join(lines).strip()
    if "gradient" not in keywords.lower().split():
        keywords = f"gradient {keywords}".strip()
    if "conp" not in keywords.lower().split():
        keywords += "\nconp"
    match = re.search(r"\bpressure\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*GPa\b", keywords, re.I)
    if "target_pressure_gpa" in options:
        pressure = float(options["target_pressure_gpa"])
    elif match:
        pressure = float(match.group(1))
    else:
        pressure = 0.0
        keywords += "\npressure 0 GPa"
    return keywords, pressure


def _lbfgs_direction(gradient: np.ndarray, history: list[tuple[np.ndarray, np.ndarray]], memory: int) -> np.ndarray:
    if not history:
        return -gradient
    query = gradient.copy()
    coefficients: list[float] = []
    for step, change in reversed(history[-memory:]):
        denominator = float(np.dot(change, step))
        coefficient = float(np.dot(step, query) / denominator) if denominator > 1.0e-16 else 0.0
        coefficients.append(coefficient)
        query -= coefficient * change
    last_step, last_change = history[-1]
    denominator = float(np.dot(last_change, last_change))
    scale = float(np.dot(last_step, last_change) / denominator) if denominator > 1.0e-16 else 1.0
    direction = scale * query
    for (step, change), coefficient in zip(history[-memory:], reversed(coefficients)):
        denominator = float(np.dot(change, step))
        beta = float(np.dot(change, direction) / denominator) if denominator > 1.0e-16 else 0.0
        direction += step * (coefficient - beta)
    return -direction


def _stage_connections(atoms: Atoms, mult: float) -> str:
    tags = infer_molecule_tags_natural_cutoffs(atoms, include_periodic_bonds=True, mult=mult)
    atoms.set_tags(tags)
    connections = infer_natural_cutoff_connections(
        atoms, molecule_tag=None, same_tag_only=True, include_periodic_bonds=True,
        index_base=1, local_indexing=False, mult=mult
    )
    return "\n".join(f"connect {first} {second}" for first, second in connections)


def _project_gradient(
    asu: Atoms,
    full: Atoms,
    records,
    operations,
    groups: list[np.ndarray],
    atom_gradient: np.ndarray,
    strain: np.ndarray,
    cell: np.ndarray,
    cell_basis: np.ndarray,
    pressure_gpa: float,
    volume: float,
):
    if atom_gradient.shape != (len(full), 3):
        raise ValueError(f"GULP returned gradient shape {atom_gradient.shape}, expected {(len(full), 3)}")
    group_of_atom = np.full(len(asu), -1, dtype=int)
    for index, group in enumerate(groups):
        group_of_atom[group] = index
    centers = np.asarray([asu.get_positions()[group].mean(axis=0) for group in groups])
    molecular = np.zeros((len(groups), 6), dtype=float)
    inv_cell = np.linalg.inv(cell)
    for full_index, (asu_index, operation_index, _) in enumerate(records):
        rotation = np.asarray(operations[operation_index][0], dtype=float)
        affine = inv_cell @ rotation.T @ cell
        mapped_gradient = affine.T @ atom_gradient[full_index]
        group_index = int(group_of_atom[asu_index])
        local = asu.get_positions()[asu_index] - centers[group_index]
        molecular[group_index, :3] += mapped_gradient
        molecular[group_index, 3:] += np.cross(local, mapped_gradient)
    cell_gradient = np.asarray([float(np.sum(strain * basis)) for basis in cell_basis])
    cell_gradient += np.asarray([pressure_gpa * volume * GPA_A3_TO_EV * np.trace(basis) for basis in cell_basis])
    return (
        np.concatenate([molecular.reshape(-1), cell_gradient]),
        float(max((np.linalg.norm(item[:3]) for item in molecular), default=0.0)),
        float(max((np.linalg.norm(item[3:]) for item in molecular), default=0.0)),
        float(np.linalg.norm(cell_gradient, ord=np.inf)),
    )


def _load_asu(calc_dir: Path) -> tuple[Atoms, Atoms]:
    full_path = calc_dir.parent / "standardized_full.cif"
    asu_path = calc_dir.parent / "asymmetric_unit.xyz"
    if not full_path.exists() or not asu_path.exists():
        raise FileNotFoundError("Symmetry-constrained rigid mode requires standardized_full.cif and asymmetric_unit.xyz")
    full = read(full_path)
    asu = read(asu_path)
    for atoms in (full, asu):
        atoms.set_pbc([True, True, True])
        atoms.set_cell(full.cell)
    return asu, full


def run_rigid_gfnff_symmetry_stage(calc_dir: Path, stage: dict[str, object], gulp_command: str, spacegroup_number: int) -> dict[str, object]:
    started = time.monotonic()
    options = dict(stage.get("rigid") or {})
    symmetry = options.get("symmetry")
    if isinstance(symmetry, dict):
        merged = dict(symmetry)
        merged.update(options)
        options = merged
    number = _select_spacegroup(calc_dir, options, int(spacegroup_number))
    tolerance = float(options.get("tolerance", options.get("symmetry_tolerance", 0.05)))
    if tolerance <= 0.0:
        raise ValueError("symmetry.tolerance must be positive")
    allow_special = bool(options.get("allow_special_positions", False))
    allow_split = bool(options.get("allow_split_molecules", False))
    asu, template = _load_asu(calc_dir)
    mult = float(options.get("natural_mult", 1.1))
    tags = infer_molecule_tags_natural_cutoffs(asu, include_periodic_bonds=True, mult=mult)
    groups = [np.where(tags == tag)[0] for tag in sorted(set(tags.tolist()))]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Every independent rigid body must contain at least two atoms")
    if not allow_split and len(asu) >= len(template):
        raise ValueError("The prepared ASU is not smaller than the full cell; check symmetry preparation")

    reference = asu.copy()
    _unwrap_groups(reference, groups)
    initial_cell = np.asarray(reference.cell, dtype=float).copy()
    positions = reference.get_positions().copy()
    centers = np.asarray([positions[group].mean(axis=0) for group in groups])
    center_fractions = centers @ np.linalg.inv(initial_cell)
    operations = _symmetry_operations(template, number, tolerance)
    cell_basis = _cell_strain_basis(initial_cell, operations, str(options.get("cell_mode", "symmetry")))
    keywords, pressure = _prepare_keywords(str(stage.get("keywords") or DEFAULT_KEYWORDS), options)
    stage_options = str(stage.get("options") or DEFAULT_OPTIONS).strip()
    if "gfnff_scale" not in stage_options.lower():
        stage_options = "\n".join(part for part in (DEFAULT_OPTIONS, stage_options) if part)
    stage_options = "\n".join(
        line for line in stage_options.splitlines()
        if not line.strip().lower().startswith(("connect ", "output ", "dump "))
    ).strip()

    steps = max(1, int(options.get("steps", 100)))
    memory = max(1, int(options.get("lbfgs_memory", 7)))
    limits = np.concatenate([
        np.tile([float(options.get("translation_step", 0.01))] * 3 + [float(options.get("rotation_step", 0.001))] * 3, len(groups)),
        np.full(len(cell_basis), float(options.get("cell_step", 0.0005))),
    ])
    if np.any(limits <= 0.0):
        raise ValueError("rigid translation_step, rotation_step and cell_step must be positive")
    force_tol = float(options.get("force_tolerance", options.get("gtol", 1.0e-3)))
    torque_tol = float(options.get("torque_tolerance", options.get("gtol", 1.0e-3)))
    cell_tol = float(options.get("cell_gradient_tolerance", options.get("gtol", 1.0e-3)))
    energy_tol = float(options.get("energy_tolerance", 1.0e-8))
    patience = max(1, int(options.get("patience", 3)))
    backtrack = float(options.get("backtrack_factor", 0.5))
    if not 0.0 < backtrack < 1.0:
        raise ValueError("rigid.backtrack_factor must be between 0 and 1")

    n_molecules = len(groups)
    x = np.zeros(n_molecules * 6 + len(cell_basis))
    history: list[tuple[np.ndarray, np.ndarray]] = []
    previous_x: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    best: tuple[float, float, np.ndarray, Atoms, Path] | None = None
    stable = 0
    last_step = 0
    log_path = calc_dir / "rigid_symmetry_steps.csv"

    with log_path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=[
            "step", "energy_ev", "enthalpy_ev", "volume_a3", "max_force", "max_torque",
            "cell_gradient", "accepted", "spacegroup", "n_asu_molecules",
        ])
        writer.writeheader()
        for step in range(1, steps + 1):
            last_step = step
            molecule_parameters = x[:n_molecules * 6].reshape(n_molecules, 6)
            rotations = np.asarray([_exp_so3(item[3:]) for item in molecule_parameters])
            strain = sum(value * basis for value, basis in zip(x[-len(cell_basis):], cell_basis))
            cell = initial_cell @ _exp_symmetric(strain)
            asu_current = _apply_asu_parameters(reference, groups, molecule_parameters[:, :3], rotations, cell, centers, center_fractions)
            full, records, special = _expand_asu(asu_current, groups, operations, tolerance, allow_special)
            if len(full) != len(template):
                raise ValueError(f"Symmetry expansion produced {len(full)} atoms, expected {len(template)}")
            options_text = "\n".join(part for part in (stage_options, _stage_connections(full, mult)) if part)
            prefix = f"{stage['prefix']}_step_{step:04d}"
            gin = calc_dir / f"{prefix}.gin"
            got = calc_dir / f"{prefix}.got"
            gin.write_text(_render_input(full, keywords, options_text))
            if _run_gulp(gulp_command, prefix, calc_dir) != 0:
                raise RuntimeError(f"GULP symmetry-constrained step failed: {got.name}")
            raw = read_results(str(got))
            energies = raw.get("energy") or []
            atom_gradient = np.asarray(raw.get("gradient"), dtype=float)
            cell_derivative = np.asarray(raw.get("strain"), dtype=float)
            if not energies or atom_gradient.shape != (len(full), 3) or cell_derivative.shape != (3, 3):
                raise RuntimeError(f"Could not read complete pGFNFF gradients from {got.name}")
            energy = float(energies[-1])
            volume = float(raw.get("volume"))
            enthalpy = energy + pressure * volume * GPA_A3_TO_EV
            gradient, max_force, max_torque, max_cell = _project_gradient(
                asu_current, full, records, operations, groups, atom_gradient, cell_derivative,
                cell, cell_basis, pressure, volume
            )
            if best is None or enthalpy < best[0] - energy_tol:
                best = (enthalpy, energy, x.copy(), full.copy(), got)
            converged_now = max_force <= force_tol and max_torque <= torque_tol and max_cell <= cell_tol
            stable = stable + 1 if converged_now else 0
            writer.writerow({
                "step": step, "energy_ev": energy, "enthalpy_ev": enthalpy, "volume_a3": volume,
                "max_force": max_force, "max_torque": max_torque, "cell_gradient": max_cell,
                "accepted": 1, "spacegroup": number, "n_asu_molecules": n_molecules,
            })
            if stable >= patience:
                break

            if previous_x is not None and previous_gradient is not None:
                delta_x = x - previous_x
                delta_gradient = gradient - previous_gradient
                if float(np.dot(delta_x, delta_gradient)) > 1.0e-12:
                    history.append((delta_x, delta_gradient))
                    history = history[-memory:]
            previous_x = x.copy()
            previous_gradient = gradient.copy()
            direction = _lbfgs_direction(gradient, history, memory)
            direction /= max(1.0, float(np.max(np.abs(direction) / limits)))
            if float(np.dot(direction, gradient)) >= 0.0:
                direction = -gradient
                direction /= max(1.0, float(np.max(np.abs(direction) / limits)))

            trial_scale = 1.0
            accepted = False
            while trial_scale >= 1.0e-4:
                trial_x = x + trial_scale * direction
                trial_parameters = trial_x[:n_molecules * 6].reshape(n_molecules, 6)
                trial_rotations = np.asarray([_exp_so3(item[3:]) for item in trial_parameters])
                trial_strain = sum(value * basis for value, basis in zip(trial_x[-len(cell_basis):], cell_basis))
                trial_cell = initial_cell @ _exp_symmetric(trial_strain)
                trial_asu = _apply_asu_parameters(reference, groups, trial_parameters[:, :3], trial_rotations, trial_cell, centers, center_fractions)
                trial_full, _, _ = _expand_asu(trial_asu, groups, operations, tolerance, allow_special)
                trial_prefix = f"{stage['prefix']}_line_{step:04d}_{int(round(-np.log10(trial_scale))):02d}"
                trial_got = calc_dir / f"{trial_prefix}.got"
                trial_gin = calc_dir / f"{trial_prefix}.gin"
                trial_gin.write_text(_render_input(trial_full, keywords, "\n".join(part for part in (stage_options, _stage_connections(trial_full, mult)) if part)))
                if _run_gulp(gulp_command, trial_prefix, calc_dir) == 0:
                    trial_raw = read_results(str(trial_got))
                    trial_energy = trial_raw.get("energy") or []
                    try:
                        trial_volume = float(trial_raw.get("volume"))
                        trial_objective = float(trial_energy[-1]) + pressure * trial_volume * GPA_A3_TO_EV
                    except (TypeError, ValueError, IndexError):
                        trial_objective = float("inf")
                    if trial_objective < enthalpy - energy_tol:
                        accepted = True
                        break
                trial_scale *= backtrack
            if not accepted:
                break
            x = trial_x

    if best is None:
        raise RuntimeError("The symmetry-constrained rigid stage produced no valid GULP point")
    _, best_energy, best_x, _, best_got = best
    best_parameters = best_x[:n_molecules * 6].reshape(n_molecules, 6)
    best_rotations = np.asarray([_exp_so3(item[3:]) for item in best_parameters])
    best_strain = sum(value * basis for value, basis in zip(best_x[-len(cell_basis):], cell_basis))
    best_cell = initial_cell @ _exp_symmetric(best_strain)
    final_asu = _apply_asu_parameters(reference, groups, best_parameters[:, :3], best_rotations, best_cell, centers, center_fractions)
    final_full, _, special = _expand_asu(final_asu, groups, operations, tolerance, allow_special)
    prefix = str(stage["prefix"])
    final_gin = calc_dir / f"{prefix}.gin"
    final_got = calc_dir / f"{prefix}.got"
    final_cif = calc_dir / f"{prefix}.cif"
    final_asu_cif = calc_dir / f"{prefix}_asymmetric_unit.cif"
    final_options = "\n".join(part for part in (stage_options, _stage_connections(final_full, mult)) if part)
    final_gin.write_text(_render_input(final_full, keywords, final_options))
    shutil.copy2(best_got, final_got)
    write(final_cif, final_full, format="cif")
    write(final_asu_cif, final_asu, format="cif")
    (calc_dir / f"{prefix}_symmetry.json").write_text(json.dumps({
        "spacegroup_number": number, "n_asu_molecules": n_molecules,
        "n_asu_atoms": len(final_asu), "n_full_atoms": len(final_full),
        "special_positions": bool(special), "source": "symmetry_constrained_rigid_gfnff",
    }, indent=2) + "\n")
    final_raw = read_results(str(final_got))
    energies = final_raw.get("energy") or [best_energy]
    return {
        "gulp_status": "rigid_symmetry_converged" if stable >= patience else "rigid_symmetry_steps_completed",
        "converged": stable >= patience,
        "energy_final_ev": float(energies[-1]), "volume": float(final_raw.get("volume")),
        "runtime_seconds": time.monotonic() - started, "n_atoms_irreducible": len(final_asu),
        "n_atoms_total": len(final_full), "gin": final_gin.name, "got": final_got.name,
        "cif": final_cif.name, "rigid_scope": "asu_symmetry", "rigid_steps_completed": last_step,
        "rigid_resymmetrized": True, "rigid_spacegroup_number": number,
        "resymmetrization_status": "strict_during_optimization",
        "message": f"optimized {n_molecules} independent rigid molecules in space group {number}",
    }


def build_symmetric_stage_input(
    calc_dir: Path,
    source_prefix: str,
    stage: dict[str, object],
    spacegroup_number: int,
    library_name: str | None = None,
) -> str:
    asu_path = calc_dir / f"{source_prefix}_asymmetric_unit.cif"
    if not asu_path.exists():
        raise FileNotFoundError(f"Rigid ASU file not found: {asu_path.name}")
    asu = read(asu_path)
    asu.set_pbc([True, True, True])
    tags = infer_molecule_tags_natural_cutoffs(asu, include_periodic_bonds=True, mult=1.1)
    asu.set_tags(tags)
    connections = infer_natural_cutoff_connections(
        asu, molecule_tag=None, same_tag_only=True, include_periodic_bonds=True,
        index_base=1, local_indexing=False, mult=1.1
    )
    options = "\n".join(
        line for line in str(stage.get("options") or "").splitlines()
        if not line.strip().lower().startswith(("connect ", "output ", "dump ", "spacegroup "))
    ).strip()
    generated = "\n".join(part for part in (
        options,
        "\n".join(f"connect {first} {second}" for first, second in connections),
        f"output movie cif {stage['prefix']}.cif",
        f"dump {stage['prefix']}.grs" if bool(stage.get("needs_restart")) else "",
    ) if part)
    if library_name and "reaxff" in str(stage.get("keywords", "")).lower().split():
        generated = f"library {library_name}\n{generated}"
    keywords = f"{stage['keywords'].rstrip()}\nspacegroup {int(spacegroup_number)}"
    return _render_input(asu, keywords, generated)
