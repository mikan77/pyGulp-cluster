from __future__ import annotations

import json
import csv
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from pygulp.io.read_gulp import read_results
from pygulp.molecule.connections import (
    infer_molecule_tags_natural_cutoffs,
    infer_natural_cutoff_connections,
)
from pygulp.stages import parse_got


DEFAULT_KEYWORDS = "gradient conp conse qok c6 gfnff gwolf noauto"
DEFAULT_OPTIONS = "gfnff_scale 0.8 1.343 0.727 1.0 2.859\nmaths mrrr"
GPA_A3_TO_EV = 0.006241509074


def _exp_so3(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1.0e-14:
        return np.eye(3)
    axis = vector / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _match_asu_to_full(asu: Atoms, full: Atoms, tolerance: float) -> list[int]:
    asu_scaled = asu.get_scaled_positions(wrap=True)
    full_scaled = full.get_scaled_positions(wrap=True)
    unused = set(range(len(full)))
    mapping: list[int] = []
    lattice = np.asarray(full.cell)

    for symbol, position in zip(asu.get_chemical_symbols(), asu_scaled):
        candidates = []
        for index in unused:
            if full[index].symbol != symbol:
                continue
            delta = position - full_scaled[index]
            delta -= np.round(delta)
            distance = float(np.linalg.norm(delta @ lattice))
            if distance <= tolerance:
                candidates.append((distance, index))
        if not candidates:
            raise ValueError("Could not map asymmetric-unit atoms to standardized full cell")
        _, selected = min(candidates)
        mapping.append(selected)
        unused.remove(selected)
    return mapping


def _unwrap_groups(atoms: Atoms, groups: list[np.ndarray]) -> None:
    cell = np.asarray(atoms.cell)
    scaled = atoms.get_scaled_positions(wrap=False)
    positions = atoms.get_positions()
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


def _apply_rigid_transform(
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
        center = center_fractions[group_index] @ cell
        local_positions = reference.get_positions()[indices] - reference_centers[group_index]
        positions[indices] = local_positions @ rotations[group_index].T + center
        positions[indices] += translations[group_index]
    atoms.set_cell(cell)
    atoms.set_positions(positions)
    return atoms


def _expand_asu(asu: Atoms, spacegroup_number: int) -> Atoms:
    if int(spacegroup_number) == 1:
        return asu.copy()
    from pymatgen.symmetry.groups import SpaceGroup

    spacegroup = SpaceGroup.from_int_number(int(spacegroup_number))
    scaled = asu.get_scaled_positions(wrap=True)
    expanded_symbols: list[str] = []
    expanded_positions: list[np.ndarray] = []
    lattice = np.asarray(asu.cell)
    for operation in spacegroup.symmetry_ops:
        transformed = scaled @ operation.rotation_matrix.T + operation.translation_vector
        for symbol, position in zip(asu.get_chemical_symbols(), np.mod(transformed, 1.0)):
            duplicate = False
            for old_symbol, old_position in zip(expanded_symbols, expanded_positions):
                if old_symbol != symbol:
                    continue
                delta = position - old_position
                delta -= np.round(delta)
                if float(np.linalg.norm(delta @ lattice)) <= 1.0e-5:
                    duplicate = True
                    break
            if not duplicate:
                expanded_symbols.append(symbol)
                expanded_positions.append(position)
    return Atoms(
        symbols=expanded_symbols,
        cell=asu.cell,
        scaled_positions=expanded_positions,
        pbc=True,
    )


def _render_input(
    atoms: Atoms,
    keywords: str,
    options: str,
    spacegroup_number: int,
    library_name: str | None = None,
) -> str:
    lines = [keywords.rstrip(), "title", "pyGulp rigid GFNFF", "end", ""]
    lines.append("cell")
    lines.append("{:9.6f} {:9.6f} {:9.6f} {:8.5f} {:8.5f} {:8.5f}".format(*atoms.cell.cellpar()))
    lines.append("frac")
    charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    if len(charges) != len(atoms):
        charges = np.zeros(len(atoms), dtype=float)
    for symbol, position, charge in zip(
        atoms.get_chemical_symbols(), atoms.get_scaled_positions(wrap=False), charges
    ):
        lines.append(
            f" {symbol:<2} {position[0]:10.7f}  {position[1]:10.7f}  "
            f"{position[2]:10.7f}  {charge:10.5f}"
        )
    lines.extend(["", "spacegroup", str(int(spacegroup_number)), ""])
    if library_name:
        lines.append(f"library {library_name}")
    if options.strip():
        lines.append(options.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def _run_gulp(command_template: str, prefix: str, calc_dir: Path) -> int:
    command = command_template.replace("PREFIX", prefix)
    completed = subprocess.run(command, cwd=calc_dir, shell=True, check=False)
    return int(completed.returncode)


def _select_scope(
    calc_dir: Path,
    options: dict[str, object],
    default_spacegroup_number: int = 1,
) -> tuple[Atoms, list[np.ndarray], int, str]:
    work_dir = calc_dir.parent
    full = read(work_dir / "standardized_full.cif")
    full.set_pbc([True, True, True])
    full_tags = infer_molecule_tags_natural_cutoffs(full, include_periodic_bonds=True, mult=1.1)
    full_groups = [np.where(full_tags == tag)[0] for tag in sorted(set(full_tags.tolist()))]

    scope = str(options.get("scope", "auto")).strip().lower()
    if scope not in {"auto", "asu", "full_cell"}:
        raise ValueError("rigid.scope must be one of: auto, asu, full_cell")

    asu_path = work_dir / "asymmetric_unit.xyz"
    symmetry_path = work_dir / "symmetry.json"
    if not asu_path.exists() or not symmetry_path.exists():
        if scope == "asu":
            raise ValueError("rigid.scope=asu requires asymmetric_unit.xyz and symmetry.json")
        return full, full_groups, 1, "full_cell"

    asu = read(asu_path)
    asu.set_cell(full.cell)
    asu.set_pbc([True, True, True])
    metadata = json.loads(symmetry_path.read_text())
    spacegroup_number = int(metadata.get("gulp_spacegroup_number", default_spacegroup_number))
    mapping = _match_asu_to_full(asu, full, tolerance=0.05)
    asu_set = set(mapping)
    asu_tags = {int(full_tags[index]) for index in mapping}
    groups_by_tag = {int(tag): set(group.tolist()) for tag, group in zip(sorted(set(full_tags.tolist())), full_groups)}
    complete = all(groups_by_tag[tag].issubset(asu_set) for tag in asu_tags)

    if scope == "asu" and not complete:
        raise ValueError("rigid.scope=asu requested, but ASU contains only fragments of a molecule")
    if scope == "auto" and not complete:
        return full, full_groups, 1, "full_cell"

    selected = full[mapping]
    selected.set_cell(full.cell)
    selected.set_pbc([True, True, True])
    local_tags = np.asarray([int(full_tags[index]) for index in mapping])
    unique_tags = {tag: number for number, tag in enumerate(sorted(set(local_tags.tolist())))}
    selected.set_tags([unique_tags[int(tag)] for tag in local_tags])
    groups = [np.where(np.asarray(selected.get_tags()) == tag)[0] for tag in sorted(set(selected.get_tags().tolist()))]
    return selected, groups, spacegroup_number, "asu"


def _resymmetrize_full_cell(
    full: Atoms,
    groups: list[np.ndarray],
    calc_dir: Path,
    prefix: str,
    options: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "rigid_resymmetrized": False,
        "rigid_spacegroup_number": 1,
    }
    if not bool(options.get("resymmetrize_after", True)):
        result["resymmetrization_status"] = "disabled"
        return result

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        symprec = float(options.get("symmetry_tolerance", 0.01))
        angle_tolerance = float(options.get("angle_tolerance", 5.0))
        structure = AseAtomsAdaptor.get_structure(full)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        spacegroup_number = int(analyzer.get_space_group_number())
        result["rigid_spacegroup_number"] = spacegroup_number
        result["rigid_spacegroup_symbol"] = str(analyzer.get_space_group_symbol())
        if spacegroup_number == 1:
            result["resymmetrization_status"] = "p1_only"
            return result

        symmetrized = analyzer.get_symmetrized_structure()
        unique_indices = [int(indices[0]) for indices in symmetrized.equivalent_indices]
        asu_structure = symmetrized.structure[unique_indices]
        symmetrized_full = AseAtomsAdaptor.get_atoms(symmetrized.structure)
        asu = AseAtomsAdaptor.get_atoms(asu_structure)
        symmetrized_full.set_pbc([True, True, True])
        asu.set_pbc([True, True, True])

        if len(symmetrized_full) != len(full):
            result["resymmetrization_status"] = "atom_count_changed"
            return result

        mapped = _match_asu_to_full(asu, full, tolerance=max(0.05, symprec * 5.0))
        asu_set = set(mapped)
        if bool(options.get("require_molecular_consistency", True)):
            for group in groups:
                group_set = set(int(index) for index in group)
                if group_set.intersection(asu_set) and not group_set.issubset(asu_set):
                    result["resymmetrization_status"] = "molecule_split"
                    return result

        full_path = calc_dir / f"{prefix}_resymmetrized_full.cif"
        asu_path = calc_dir / f"{prefix}_resymmetrized_asymmetric_unit.cif"
        symmetry_path = calc_dir / f"{prefix}_resymmetrized_symmetry.json"
        write(full_path, symmetrized_full, format="cif")
        write(asu_path, asu, format="cif")
        symmetry_path.write_text(
            json.dumps(
                {
                    "gulp_spacegroup_number": spacegroup_number,
                    "spacegroup_symbol": str(analyzer.get_space_group_symbol()),
                    "symprec": symprec,
                    "angle_tolerance": angle_tolerance,
                    "source": "rigid_resymmetrization",
                },
                indent=2,
            )
            + "\n"
        )
        result.update(
            {
                "rigid_resymmetrized": True,
                "resymmetrization_status": "success",
                "resymmetrized_full_cif": full_path.name,
                "resymmetrized_asu_cif": asu_path.name,
                "resymmetrized_symmetry_json": symmetry_path.name,
                "rigid_spacegroup_number": spacegroup_number,
            }
        )
        return result
    except Exception as exc:
        result["resymmetrization_status"] = "failed"
        result["resymmetrization_error"] = repr(exc)
        return result


def build_resymmetrized_stage_input(
    calc_dir: Path,
    stage: dict[str, object],
    spacegroup_number: int,
    library_name: str | None = None,
) -> str:
    asu_path = calc_dir / f"{stage['prefix'].replace('_' + str(stage['name']).split('_', 1)[-1], '')}_resymmetrized_asymmetric_unit.cif"
    candidates = sorted(calc_dir.glob("*_resymmetrized_asymmetric_unit.cif"))
    if candidates:
        asu_path = candidates[-1]
    if not asu_path.exists():
        raise FileNotFoundError(f"Resymmetrized ASU file not found in {calc_dir}")

    asu = read(asu_path)
    asu.set_pbc([True, True, True])
    tags = infer_molecule_tags_natural_cutoffs(asu, include_periodic_bonds=True, mult=1.1)
    asu.set_tags(tags)
    connections = infer_natural_cutoff_connections(
        asu,
        molecule_tag=None,
        same_tag_only=True,
        include_periodic_bonds=True,
        index_base=1,
        local_indexing=False,
        mult=1.1,
    )
    stage_options = "\n".join(
        line
        for line in str(stage.get("options") or "").splitlines()
        if not (line.strip() and line.split()[0].lower() == "connect")
    ).strip()
    generated = "\n".join(
        part
        for part in (
            stage_options,
            "\n".join(f"connect {first} {second}" for first, second in connections),
            f"output movie cif {stage['prefix']}.cif",
            f"dump {stage['prefix']}.grs" if bool(stage.get("needs_restart")) else "",
        )
        if part
    )
    if "reaxff" in str(stage.get("keywords", "")).lower().split() and library_name:
        generated = f"library {library_name}\n{generated}"
    return _render_input(
        asu,
        str(stage["keywords"]),
        generated,
        int(spacegroup_number),
    )


def run_rigid_gfnff_stage(
    calc_dir: Path,
    stage: dict[str, object],
    gulp_command: str,
    spacegroup_number: int,
) -> dict[str, object]:
    options = dict(stage.get("rigid") or {})
    cell_mode = str(options.get("cell_mode", "isotropic")).strip().lower()
    if cell_mode != "isotropic":
        raise ValueError(
            "rigid_gfnff now requires rigid.cell_mode=isotropic; fixed-cell mode has been removed"
        )

    atoms, groups, detected_spacegroup, scope = _select_scope(
        calc_dir,
        options,
        default_spacegroup_number=spacegroup_number,
    )
    if scope == "full_cell":
        detected_spacegroup = 1

    keywords = str(stage.get("keywords") or DEFAULT_KEYWORDS).strip()
    keyword_words = keywords.lower().split()
    if "conv" in keyword_words:
        raise ValueError("rigid_gfnff requires conp; conv/fixed-cell mode is no longer supported")
    if "conp" not in keyword_words:
        keywords = f"{keywords}\nconp"
        keyword_words = keywords.lower().split()
    if "gradient" not in keywords.lower().split():
        keywords = f"gradient {keywords}"

    pressure_match = re.search(
        r"\bpressure\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*GPa\b",
        keywords,
        re.IGNORECASE,
    )
    if "target_pressure_gpa" in options:
        target_pressure_gpa = float(options["target_pressure_gpa"])
    elif pressure_match:
        target_pressure_gpa = float(pressure_match.group(1))
    else:
        target_pressure_gpa = 0.0
        keywords = f"{keywords}\npressure 0 GPa"

    stage_options = "\n".join(
        line
        for line in str(stage.get("options") or "").splitlines()
        if not (line.strip() and line.split()[0].lower() == "connect")
    ).strip()
    if "gfnff_scale" not in stage_options.lower():
        stage_options = "\n".join(part for part in (DEFAULT_OPTIONS, stage_options) if part)
    connections = infer_natural_cutoff_connections(
        atoms,
        molecule_tag=None,
        same_tag_only=True,
        include_periodic_bonds=True,
        index_base=1,
        local_indexing=False,
        mult=float(options.get("natural_mult", 1.1)),
    )
    connection_text = "\n".join(f"connect {first} {second}" for first, second in connections)
    input_options = "\n".join(part for part in (stage_options, connection_text) if part)

    steps = max(1, int(options.get("steps", 50)))
    translation_step = float(options.get("translation_step", 0.008))
    rotation_step = float(options.get("rotation_step", 0.00005))
    force_tolerance = float(options.get("force_tolerance", 0.0))
    torque_tolerance = float(options.get("torque_tolerance", 0.0))
    cell_gradient_tolerance = float(options.get("cell_gradient_tolerance", 0.0))
    energy_tolerance = float(options.get("energy_tolerance", 0.0))
    patience = max(1, int(options.get("patience", 5)))
    cell_step = float(options.get("cell_step", 5.0e-6))
    max_cell_change = abs(float(options.get("max_cell_change", 0.01)))
    backtrack_factor = float(options.get("backtrack_factor", 0.5))
    objective_tolerance = float(options.get("objective_tolerance", 1.0e-8))
    if cell_step <= 0.0 or max_cell_change <= 0.0:
        raise ValueError("rigid.cell_step and rigid.max_cell_change must be positive")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("rigid.backtrack_factor must be between 0 and 1")

    reference = atoms.copy()
    _unwrap_groups(reference, groups)
    reference_cell = np.asarray(reference.cell, dtype=float).copy()
    reference_positions = reference.get_positions().copy()
    reference_centers = np.array(
        [reference_positions[group].mean(axis=0) for group in groups],
        dtype=float,
    )
    center_fractions = reference_centers @ np.linalg.inv(reference_cell)
    cell = reference_cell.copy()
    translations = np.zeros((len(groups), 3), dtype=float)
    rotations = np.repeat(np.eye(3)[None, :, :], len(groups), axis=0)
    best_energy: float | None = None
    best_objective: float | None = None
    best_step = 0
    best_atoms: Atoms | None = None
    best_got: Path | None = None
    previous_energy: float | None = None
    stable_steps = 0
    accepted_energy: float | None = None
    accepted_objective: float | None = None
    accepted_translations = translations.copy()
    accepted_rotations = rotations.copy()
    accepted_cell = cell.copy()
    converged = False
    last_data: dict[str, object] = {}
    step_log_path = calc_dir / "rigid_steps.csv"
    step_log = step_log_path.open("w", newline="")
    step_writer = csv.DictWriter(
        step_log,
        fieldnames=[
            "step",
            "energy_ev",
            "enthalpy_ev",
            "volume_a3",
            "max_force",
            "max_torque",
            "cell_gradient",
            "cell_scale",
            "accepted",
        ],
    )
    step_writer.writeheader()

    for step in range(1, steps + 1):
        current = _apply_rigid_transform(
            reference,
            groups,
            translations,
            rotations,
            cell,
            reference_centers,
            center_fractions,
        )
        iteration_prefix = f"{stage['prefix']}_step_{step:04d}"
        gin_path = calc_dir / f"{iteration_prefix}.gin"
        got_path = calc_dir / f"{iteration_prefix}.got"
        gin_path.write_text(_render_input(current, keywords, input_options, detected_spacegroup))
        returncode = _run_gulp(gulp_command, iteration_prefix, calc_dir)
        if returncode != 0:
            step_log.close()
            raise RuntimeError(f"GULP rigid iteration returned exit code {returncode}")

        raw = read_results(str(got_path))
        energies = raw.get("energy") or []
        gradient = np.asarray(raw.get("gradient"), dtype=float)
        cell_gradient_tensor = np.asarray(raw.get("strain"), dtype=float)
        if (
            not energies
            or gradient.ndim != 2
            or gradient.shape[0] != len(current)
            or cell_gradient_tensor.shape != (3, 3)
            or not np.all(np.isfinite(cell_gradient_tensor))
        ):
            step_log.close()
            raise RuntimeError(f"Could not read rigid GFNFF gradients from {got_path.name}")

        try:
            volume = float(raw.get("volume"))
        except (TypeError, ValueError):
            step_log.close()
            raise RuntimeError(f"Could not read cell volume from {got_path.name}")
        energy = float(energies[-1])
        objective = energy + target_pressure_gpa * volume * GPA_A3_TO_EV
        cell_gradient_tensor = 0.5 * (cell_gradient_tensor + cell_gradient_tensor.T)
        cell_gradient = float(np.trace(cell_gradient_tensor) / 3.0)

        inverse_cell = np.linalg.inv(np.asarray(current.cell))
        forces = -(gradient @ inverse_cell)
        molecule_forces = np.array([np.sum(forces[group], axis=0) for group in groups])
        current_positions = current.get_positions()
        centers = np.array([current_positions[group].mean(axis=0) for group in groups])
        torques = np.array(
            [
                np.sum(np.cross(current_positions[group] - center, forces[group]), axis=0)
                for group, center in zip(groups, centers)
            ]
        )
        max_force = float(max(np.linalg.norm(value) for value in molecule_forces))
        max_torque = float(max(np.linalg.norm(value) for value in torques))

        accepted = accepted_objective is None or objective <= accepted_objective + objective_tolerance
        if not accepted:
            translations = accepted_translations.copy()
            rotations = accepted_rotations.copy()
            cell = accepted_cell.copy()
            translation_step *= backtrack_factor
            rotation_step *= backtrack_factor
            cell_step *= backtrack_factor
        else:
            accepted_energy = energy
            accepted_objective = objective
            accepted_translations = translations.copy()
            accepted_rotations = rotations.copy()
            accepted_cell = cell.copy()
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_energy = energy
                best_step = step
                best_atoms = current.copy()
                best_got = got_path

            last_data = {
                "energy_initial_ev": energy,
                "energy_final_ev": energy,
                "volume": volume,
                "steps_completed": step,
                "max_force": max_force,
                "max_torque": max_torque,
                "cell_gradient": cell_gradient,
                "cell_mode": cell_mode,
                "target_pressure_gpa": target_pressure_gpa,
            }

            if previous_energy is not None and energy_tolerance > 0.0:
                stable_steps = stable_steps + 1 if abs(energy - previous_energy) <= energy_tolerance else 0
            previous_energy = energy
            cell_ok = (
                cell_gradient_tolerance > 0.0
                and abs(cell_gradient) <= cell_gradient_tolerance
            )
            if (
                force_tolerance > 0.0
                and torque_tolerance > 0.0
                and cell_ok
                and max_force <= force_tolerance
                and max_torque <= torque_tolerance
            ) or stable_steps >= patience:
                converged = True

        cell_scale = 1.0
        if accepted and not converged:
            translations += translation_step * molecule_forces
            for index, torque in enumerate(torques):
                rotations[index] = _exp_so3(rotation_step * torque) @ rotations[index]

            cell_change = float(np.clip(cell_step * cell_gradient, -max_cell_change, max_cell_change))
            cell_scale = 1.0 - cell_change
            if cell_scale <= 0.0:
                step_log.close()
                raise RuntimeError("Rigid cell update produced a non-positive cell scale")
            cell = cell_scale * cell

        step_writer.writerow(
            {
                "step": step,
                "energy_ev": energy,
                "enthalpy_ev": objective,
                "volume_a3": volume,
                "max_force": max_force,
                "max_torque": max_torque,
                "cell_gradient": cell_gradient,
                "cell_scale": cell_scale,
                "accepted": accepted,
            }
        )
        step_log.flush()
        if converged:
            break

    step_log.close()

    if best_atoms is None or best_got is None or best_energy is None:
        raise RuntimeError("Rigid GFNFF stage produced no usable energy")

    final_gin = calc_dir / f"{stage['prefix']}.gin"
    final_got = calc_dir / f"{stage['prefix']}.got"
    final_cif = calc_dir / f"{stage['prefix']}.cif"
    final_gin.write_text(_render_input(best_atoms, keywords, input_options, detected_spacegroup))
    shutil.copy2(best_got, final_got)
    output_atoms = _expand_asu(best_atoms, detected_spacegroup) if scope == "asu" else best_atoms
    write(final_cif, output_atoms, format="cif")

    resymmetrization = _resymmetrize_full_cell(
        best_atoms,
        groups,
        calc_dir,
        str(stage["prefix"]),
        options,
    ) if scope == "full_cell" else {
        "rigid_resymmetrized": False,
        "resymmetrization_status": "already_asu",
        "rigid_spacegroup_number": detected_spacegroup,
    }

    parsed = parse_got(final_got)
    last_data.update(
        {
            "energy_initial_ev": parsed.get("energy_initial_ev"),
            "energy_final_ev": parsed.get("energy_final_ev"),
            "volume": parsed.get("volume"),
            "gulp_status": "rigid_converged" if converged else "rigid_steps_completed",
            "converged": converged,
            "rigid_scope": scope,
            "rigid_best_step": best_step,
            "rigid_steps_completed": int(last_data.get("steps_completed", steps)),
        }
    )
    last_data.update(resymmetrization)
    return last_data
