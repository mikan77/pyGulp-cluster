from __future__ import annotations

import json
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
) -> Atoms:
    atoms = reference.copy()
    positions = reference.get_positions().copy()
    for group_index, group in enumerate(groups):
        indices = np.asarray(group, dtype=int)
        center = positions[indices].mean(axis=0)
        positions[indices] = (positions[indices] - center) @ rotations[group_index].T + center
        positions[indices] += translations[group_index]
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


def run_rigid_gfnff_stage(
    calc_dir: Path,
    stage: dict[str, object],
    gulp_command: str,
    spacegroup_number: int,
) -> dict[str, object]:
    options = dict(stage.get("rigid") or {})
    atoms, groups, detected_spacegroup, scope = _select_scope(
        calc_dir,
        options,
        default_spacegroup_number=spacegroup_number,
    )
    if scope == "full_cell":
        detected_spacegroup = 1

    keywords = str(stage.get("keywords") or DEFAULT_KEYWORDS).strip()
    if "gradient" not in keywords.lower().split():
        keywords = f"gradient {keywords}"
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
    energy_tolerance = float(options.get("energy_tolerance", 0.0))
    patience = max(1, int(options.get("patience", 5)))

    reference = atoms.copy()
    _unwrap_groups(reference, groups)
    translations = np.zeros((len(groups), 3), dtype=float)
    rotations = np.repeat(np.eye(3)[None, :, :], len(groups), axis=0)
    best_energy: float | None = None
    best_step = 0
    best_atoms: Atoms | None = None
    best_got: Path | None = None
    previous_energy: float | None = None
    stable_steps = 0
    converged = False
    last_data: dict[str, object] = {}

    for step in range(1, steps + 1):
        current = _apply_rigid_transform(reference, groups, translations, rotations)
        iteration_prefix = f"{stage['prefix']}_step_{step:04d}"
        gin_path = calc_dir / f"{iteration_prefix}.gin"
        got_path = calc_dir / f"{iteration_prefix}.got"
        gin_path.write_text(_render_input(current, keywords, input_options, detected_spacegroup))
        returncode = _run_gulp(gulp_command, iteration_prefix, calc_dir)
        if returncode != 0:
            raise RuntimeError(f"GULP rigid iteration returned exit code {returncode}")

        raw = read_results(str(got_path))
        energies = raw.get("energy") or []
        gradient = np.asarray(raw.get("gradient"), dtype=float)
        if not energies or gradient.ndim != 2 or gradient.shape[0] != len(current):
            raise RuntimeError(f"Could not read rigid GFNFF gradient from {got_path.name}")
        energy = float(energies[-1])
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_step = step
            best_atoms = current.copy()
            best_got = got_path

        inverse_cell = np.linalg.inv(np.asarray(current.cell))
        forces = -(gradient @ inverse_cell)
        molecule_forces = np.array([np.sum(forces[group], axis=0) for group in groups])
        centers = np.array([current.get_positions()[group].mean(axis=0) for group in groups])
        torques = np.array(
            [np.sum(np.cross(current.get_positions()[group] - center, forces[group]), axis=0) for group, center in zip(groups, centers)]
        )
        max_force = float(max(np.linalg.norm(value) for value in molecule_forces))
        max_torque = float(max(np.linalg.norm(value) for value in torques))
        last_data = {
            "energy_initial_ev": energy,
            "energy_final_ev": energy,
            "volume": raw.get("volume"),
            "steps_completed": step,
            "max_force": max_force,
            "max_torque": max_torque,
        }

        if previous_energy is not None and energy_tolerance > 0.0:
            stable_steps = stable_steps + 1 if abs(energy - previous_energy) <= energy_tolerance else 0
        previous_energy = energy
        if (
            force_tolerance > 0.0
            and torque_tolerance > 0.0
            and max_force <= force_tolerance
            and max_torque <= torque_tolerance
        ) or stable_steps >= patience:
            converged = True
            break

        translations += translation_step * molecule_forces
        for index, torque in enumerate(torques):
            rotations[index] = _exp_so3(rotation_step * torque) @ rotations[index]

    if best_atoms is None or best_got is None or best_energy is None:
        raise RuntimeError("Rigid GFNFF stage produced no usable energy")

    final_gin = calc_dir / f"{stage['prefix']}.gin"
    final_got = calc_dir / f"{stage['prefix']}.got"
    final_cif = calc_dir / f"{stage['prefix']}.cif"
    final_gin.write_text(_render_input(best_atoms, keywords, input_options, detected_spacegroup))
    shutil.copy2(best_got, final_got)
    output_atoms = _expand_asu(best_atoms, detected_spacegroup) if scope == "asu" else best_atoms
    write(final_cif, output_atoms, format="cif")

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
    return last_data
