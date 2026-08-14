from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list

from pygulp.molecule.connections import (
    infer_molecule_tags_natural_cutoffs,
    infer_natural_cutoff_connections,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    keywords: str
    options: str
    require_convergence: bool = True
    validate_atom_counts: bool = False
    mode: str = "gulp"
    cell_mode: str | None = None
    symmetry_mode: str = "auto"
    rigid_options: dict[str, object] = field(default_factory=dict)
    final_symmetry: dict[str, object] = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return self.name

    @property
    def is_optimisation(self) -> bool:
        if self.mode == "rigid_gfnff_symmetry":
            return True
        words = set(re.findall(r"[A-Za-z_]+", self.keywords.lower()))
        return bool(words & {"opti", "optimise", "optimize"})


STAGE_RESULT_FIELDS = (
    "stage",
    "name",
    "status",
    "gulp_status",
    "converged",
    "energy_initial_ev",
    "energy_final_ev",
    "volume",
    "gnorm",
    "runtime_seconds",
    "n_atoms_irreducible",
    "n_atoms_total",
    "symmetry_mode",
    "spacegroup_number",
    "gin",
    "got",
    "cif",
    "restart",
    "rigid_steps_completed",
    "structure_validation",
    "n_molecules",
    "n_isolated_atoms",
    "message",
)


def sanitize_stage_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Stage name must contain at least one letter or digit")
    return cleaned


def load_stage_specs(path: Path, default_options: str) -> list[StageSpec]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("Reading --stages-file requires PyYAML") from exc

    payload = yaml.safe_load(path.read_text())
    entries = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Stages file must contain a non-empty 'stages' list")

    stages: list[StageSpec] = []
    used_names: set[str] = set()
    default_validate_atom_counts = payload.get("validate_atom_counts", False)
    if not isinstance(default_validate_atom_counts, bool):
        raise ValueError("Top-level field 'validate_atom_counts' must be true or false")

    final_symmetry = payload.get("final_symmetry", {})
    if not isinstance(final_symmetry, dict):
        raise ValueError("Top-level field 'final_symmetry' must be a mapping")
    final_symmetry = dict(final_symmetry)
    required = final_symmetry.get("required", True)
    if not isinstance(required, bool):
        raise ValueError("final_symmetry.required must be true or false")
    try:
        final_symprec = float(final_symmetry.get("symprec", 0.05))
        final_angle_tolerance = float(final_symmetry.get("angle_tolerance", 5.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("final_symmetry tolerances must be numbers") from exc
    if final_symprec <= 0.0 or final_angle_tolerance <= 0.0:
        raise ValueError("final_symmetry tolerances must be positive")
    final_symmetry = {
        "required": required,
        "symprec": final_symprec,
        "angle_tolerance": final_angle_tolerance,
    }

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Stage #{index} must be a mapping")
        raw_name = entry.get("name")
        mode = str(entry.get("mode", "gulp")).strip().lower()
        if mode not in {"gulp", "rigid_gfnff_symmetry"}:
            raise ValueError(f"Stage #{index} has unsupported mode: {mode}")
        raw_symmetry_mode = entry.get("symmetry_mode", "auto")
        if isinstance(raw_symmetry_mode, bool):
            symmetry_mode = "auto" if raw_symmetry_mode else "off"
        else:
            symmetry_mode = str(raw_symmetry_mode).strip().lower()
        if symmetry_mode not in {"auto", "off"}:
            raise ValueError(f"Stage #{index} symmetry_mode must be auto or off")
        if mode == "rigid_gfnff_symmetry" and symmetry_mode == "off":
            raise ValueError(f"Stage #{index} rigid_gfnff_symmetry requires symmetry_mode: auto")
        keywords = entry.get("keywords")
        if mode == "rigid_gfnff_symmetry" and keywords is None:
            keywords = "gradient conp conse qok c6 gfnff gwolf noauto"
        options = entry.get("options", default_options)
        cell_mode = entry.get("cell_mode")
        if cell_mode is not None:
            cell_mode = str(cell_mode).strip().lower()
            if cell_mode not in {"conv", "conp"}:
                raise ValueError(f"Stage #{index} field 'cell_mode' must be conv or conp")
        require_convergence = entry.get("require_convergence", False if mode == "rigid_gfnff_symmetry" else True)
        validate_atom_counts = entry.get("validate_atom_counts", default_validate_atom_counts)
        rigid_options = entry.get("rigid", {})
        symmetry_options = entry.get("symmetry", {})
        if not isinstance(raw_name, str) or not isinstance(keywords, str) or not keywords.strip():
            raise ValueError(f"Stage #{index} requires non-empty string fields 'name' and 'keywords'")
        if not isinstance(options, str):
            raise ValueError(f"Stage #{index} field 'options' must be a string")
        if not isinstance(require_convergence, bool):
            raise ValueError(f"Stage #{index} field 'require_convergence' must be true or false")
        if not isinstance(validate_atom_counts, bool):
            raise ValueError(f"Stage #{index} field 'validate_atom_counts' must be true or false")
        if not isinstance(rigid_options, dict):
            raise ValueError(f"Stage #{index} field 'rigid' must be a mapping")
        if not isinstance(symmetry_options, dict):
            raise ValueError(f"Stage #{index} field 'symmetry' must be a mapping")
        rigid_options = dict(rigid_options)
        if symmetry_options:
            rigid_options["symmetry"] = dict(symmetry_options)
        cleaned_name = sanitize_stage_name(raw_name)
        if cleaned_name in used_names:
            raise ValueError(f"Duplicate stage name: {raw_name}")
        used_names.add(cleaned_name)
        keyword_words = set(re.findall(r"[A-Za-z_]+", keywords.lower()))
        has_conv = "conv" in keyword_words
        has_conp = "conp" in keyword_words
        if has_conv and has_conp:
            raise ValueError(f"Stage #{index} cannot contain both conv and conp")
        inferred_cell_mode = cell_mode
        if inferred_cell_mode is None and has_conv:
            inferred_cell_mode = "conv"
        elif inferred_cell_mode is None and has_conp:
            inferred_cell_mode = "conp"
        if inferred_cell_mode is None:
            inferred_cell_mode = None
        if inferred_cell_mode == "conv" and not has_conv:
            raise ValueError(f"Stage #{index} is declared as conv but keywords do not contain conv")
        if inferred_cell_mode == "conp" and not has_conp:
            raise ValueError(f"Stage #{index} is declared as conp but keywords do not contain conp")
        name = f"{index:02d}_{cleaned_name}"
        stages.append(
            StageSpec(
                name=name,
                keywords=keywords.rstrip(),
                options=options.rstrip(),
                require_convergence=require_convergence,
                validate_atom_counts=validate_atom_counts,
                mode=mode,
                cell_mode=inferred_cell_mode,
                symmetry_mode=symmetry_mode,
                rigid_options=dict(rigid_options),
                final_symmetry=final_symmetry,
            )
        )

    for stage in stages[:-1]:
        if not stage.is_optimisation:
            raise ValueError(f"Non-final stage '{stage.name}' must be an optimisation so GULP can write a restart")
    return stages


def parse_got(got_path: Path) -> dict[str, object]:
    data: dict[str, object] = {
        "energy_initial_ev": None,
        "energy_final_ev": None,
        "energy_is_nonprimitive": False,
        "volume": None,
        "volume_primitive": None,
        "volume_nonprimitive": None,
        "gnorm": None,
        "runtime_seconds": None,
        "n_atoms_irreducible": None,
        "n_atoms_total": None,
        "gulp_status": None,
        "completed_normally": False,
    }
    if not got_path.exists():
        return data

    energies: list[float] = []
    primitive_energies: list[float] = []
    nonprimitive_energies: list[float] = []
    volumes: list[str] = []
    runtime_seconds: float | None = None
    cpu_seconds: float | None = None
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    for line in got_path.read_text(errors="replace").splitlines():
        energy_match = re.search(
            rf"Total lattice energy\s*=\s*({number_pattern})\s*eV",
            line,
            re.I,
        )
        if energy_match:
            energies.append(float(energy_match.group(1).replace("D", "E").replace("d", "e")))
        primitive_energy_match = re.search(
            rf"Primitive unit cell\s*=\s*({number_pattern})\s*eV",
            line,
            re.I,
        )
        if primitive_energy_match:
            primitive_energies.append(
                float(primitive_energy_match.group(1).replace("D", "E").replace("d", "e"))
            )
        nonprimitive_energy_match = re.search(
            rf"Non-primitive unit cell\s*=\s*({number_pattern})\s*eV",
            line,
            re.I,
        )
        if nonprimitive_energy_match:
            nonprimitive_energies.append(
                float(nonprimitive_energy_match.group(1).replace("D", "E").replace("d", "e"))
            )
        primitive_volume_match = re.search(
            rf"Primitive cell volume\s*=\s*({number_pattern})",
            line,
            re.I,
        )
        if primitive_volume_match:
            data["volume_primitive"] = float(
                primitive_volume_match.group(1).replace("D", "E").replace("d", "e")
            )
        nonprimitive_volume_match = re.search(
            rf"Non-primitive cell volume\s*=\s*({number_pattern})",
            line,
            re.I,
        )
        if nonprimitive_volume_match:
            data["volume_nonprimitive"] = float(
                nonprimitive_volume_match.group(1).replace("D", "E").replace("d", "e")
            )
        volume_match = re.search(rf"cell volume\s*=\s*({number_pattern})", line, re.I)
        if volume_match:
            volumes.append(volume_match.group(1).replace("D", "E").replace("d", "e"))
        gnorm_match = re.search(rf"Final Gnorm\s*=\s*({number_pattern})", line, re.I)
        if gnorm_match:
            data["gnorm"] = float(gnorm_match.group(1).replace("D", "E").replace("d", "e"))
        runtime_match = re.search(
            rf"Time to end of optimisation\s*=\s*({number_pattern})\s*seconds",
            line,
            re.I,
        )
        if runtime_match:
            runtime_seconds = float(runtime_match.group(1).replace("D", "E").replace("d", "e"))
        cpu_match = re.search(rf"Total CPU time\s+({number_pattern})", line, re.I)
        if cpu_match:
            cpu_seconds = float(cpu_match.group(1).replace("D", "E").replace("d", "e"))
        irreducible_match = re.search(
            r"(?:Number of irreducible atoms/shells|Number of irreducible atoms|Number of atoms/shells in asym(?:ym)?metrical unit)\s*=\s*(\d+)",
            line,
            re.I,
        )
        if irreducible_match:
            data["n_atoms_irreducible"] = int(irreducible_match.group(1))
        total_match = re.search(
            r"(?:^\s*Total number atoms/shells\s*=|^\s*Total number of atoms(?:/shells)?\s*=|^\s*Number of atoms in unit cell\s*=|^\s*Number of atoms/shells in unit cell\s*=|^\s*Number of atoms\s*=)\s*(\d+)",
            line,
            re.I,
        )
        if total_match:
            data["n_atoms_total"] = int(total_match.group(1))
        if "Optimisation achieved" in line:
            data["gulp_status"] = "optimisation_achieved"
        elif "Maximum number of function calls has been reached" in line:
            data["gulp_status"] = "max_function_calls"
        elif "Too many failed attempts to optimise" in line:
            data["gulp_status"] = "too_many_failed_attempts"
        if "GULP has completed" in line:
            data["completed_normally"] = True

    if nonprimitive_energies:
        energies = nonprimitive_energies
        data["energy_is_nonprimitive"] = True
    elif primitive_energies:
        energies = primitive_energies
    if energies:
        data["energy_initial_ev"] = energies[0]
        data["energy_final_ev"] = energies[-1]
    if data["volume_nonprimitive"] is not None:
        data["volume"] = data["volume_nonprimitive"]
    elif data["volume_primitive"] is not None:
        data["volume"] = data["volume_primitive"]
    elif volumes:
        data["volume"] = float(volumes[-1])
    data["runtime_seconds"] = runtime_seconds if runtime_seconds is not None else cpu_seconds
    return data


def validate_got_contract(
    data: dict[str, object],
    expected_asu: int,
    expected_total: int,
    *,
    strict: bool = True,
) -> None:
    actual_asu = data.get("n_atoms_irreducible")
    actual_total = data.get("n_atoms_total")

    if actual_asu is None or actual_total is None:
        if strict:
            raise ValueError(
                "GULP atom-count lines were not found in output. "
                "Set validate_atom_counts: false in stage configuration to skip this check."
            )
        return

    actual_asu_int = int(actual_asu)
    actual_total_int = int(actual_total)

    if actual_asu_int == expected_asu and actual_total_int == expected_total:
        return

    # Some GULP runs report the primitive cell atom totals even when the input is
    # given as an ASU. In this case the irreducible count still matches and the
    # reported total is a clean divisor of the expected conventional total.
    if (
        actual_asu_int == expected_asu
        and actual_total_int > 0
        and expected_total > 0
        and actual_total_int <= expected_total
        and expected_total % actual_total_int == 0
    ):
        return

    # If both irreducible and total counts are scaled by the same factor
    # (e.g. both reduced for primitive-cell reporting), accept that as ASU mismatch.
    if (
        actual_asu_int > 0
        and expected_asu > 0
        and actual_total_int > 0
        and expected_total > 0
        and expected_asu % actual_asu_int == 0
        and expected_total % actual_total_int == 0
        and expected_asu // actual_asu_int == expected_total // actual_total_int
    ):
        return

    if actual_asu_int != expected_asu or actual_total_int != expected_total:
        raise ValueError(
            "GULP atom-count mismatch: "
            f"irreducible={actual_asu_int} (expected {expected_asu}), "
            f"total={actual_total_int} (expected {expected_total})"
        )


def _managed_option_heads(stages: list[dict[str, object]]) -> set[str]:
    heads = {"dump", "output", "pressure"}
    for stage in stages:
        for block_key in ("options", "keywords"):
            for line in str(stage.get(block_key, "")).splitlines():
                words = line.split()
                if words:
                    heads.add(words[0].lower())
    return heads


def _restart_structure(restart_text: str) -> str:
    """Return only the structural part of a GULP restart.

    GFNFF restart blocks are deliberately discarded. They are generated state,
    not portable input, and must be rebuilt by GULP for the next stage.
    """

    lines = restart_text.splitlines()
    title_index = next((i for i, line in enumerate(lines) if line.strip().lower() == "title"), None)
    if title_index is None:
        raise ValueError("GULP restart does not contain a title block")

    space_index = next(
        (i for i in range(title_index, len(lines)) if lines[i].strip().lower() in {"space", "spacegroup"}),
        None,
    )
    if space_index is None or space_index + 1 >= len(lines):
        raise ValueError("GULP restart does not contain a space-group block")

    return "\n".join(lines[title_index : space_index + 2]).rstrip()


def _structure_signature(atoms, mult: float) -> dict[str, object]:
    atoms = atoms.copy()
    atoms.set_pbc([True, True, True])
    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=mult)
    first, second, _ = neighbor_list("ijS", atoms, cutoffs)
    edges: set[tuple[int, int]] = set()
    adjacency = [set() for _ in atoms]
    for i, j in zip(first, second):
        i, j = int(i), int(j)
        if i == j:
            continue
        edge = tuple(sorted((i, j)))
        edges.add(edge)
        adjacency[i].add(j)
        adjacency[j].add(i)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(atoms)):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(component)

    signatures = []
    for component in components:
        formula = Counter(symbols[index] for index in component)
        signatures.append(
            {
                "size": len(component),
                "formula": dict(sorted(formula.items())),
            }
        )
    signatures.sort(key=lambda item: (int(item["size"]), tuple(item["formula"].items())))
    return {
        "n_atoms": len(atoms),
        "formula": dict(sorted(Counter(symbols).items())),
        "n_molecules": len(components),
        "n_edges": len(edges),
        "n_isolated_atoms": sum(not neighbours for neighbours in adjacency),
        "molecules": signatures,
    }


def validate_structure_output(
    cif_path: Path,
    expected: dict[str, object],
    mult: float,
    report_path: Path,
) -> dict[str, object]:
    if not cif_path.is_file():
        raise FileNotFoundError(f"GULP did not write structure CIF {cif_path.name}")
    actual = _structure_signature(read(cif_path), mult)
    expected_molecules = sorted(
        expected.get("molecules", []),
        key=lambda item: (int(item["size"]), tuple(item["formula"].items())),
    )
    mismatches = []
    for key in ("n_atoms", "formula", "n_molecules", "n_edges", "n_isolated_atoms"):
        if actual[key] != expected.get(key):
            mismatches.append(f"{key}: expected {expected.get(key)!r}, got {actual[key]!r}")
    if actual["molecules"] != expected_molecules:
        mismatches.append(
            f"molecules: expected {expected_molecules!r}, got {actual['molecules']!r}"
        )
    report = {"status": "valid" if not mismatches else "invalid", "expected": expected, "actual": actual, "mismatches": mismatches}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if mismatches:
        raise RuntimeError(f"Structure validation failed for {cif_path.name}: " + "; ".join(mismatches))
    return actual


def rewrite_restart(
    restart_text: str,
    stage: dict[str, object],
    managed_heads: set[str],
    library_name: str | None = None,
) -> str:
    del managed_heads
    kept = [*str(stage["keywords"]).splitlines(), _restart_structure(restart_text)]
    if library_name and not any(line.strip().lower().startswith("library ") for line in kept):
        kept.append(f"library {library_name}")
    options = str(stage.get("options", "")).rstrip()
    if options:
        kept.extend(["", options])
    kept.append(f"output movie cif {stage['prefix']}.cif")
    if bool(stage.get("needs_restart")):
        kept.append(f"dump every 1 {stage['prefix']}.grs")
    return "\n".join(kept).rstrip() + "\n"


def _without_connect_options(stage: dict[str, object]) -> dict[str, object]:
    cleaned = dict(stage)
    cleaned["options"] = "\n".join(
        line
        for line in str(stage.get("options", "")).splitlines()
        if not (line.strip() and line.split()[0].lower() == "connect")
    )
    return cleaned


def _stage_options(stage: dict[str, object], connections: str, needs_restart: bool) -> str:
    kept: list[str] = []
    skip_next = False
    for line in str(stage.get("options", "")).splitlines():
        words = line.split()
        if skip_next:
            skip_next = False
            continue
        if words and words[0].lower() in {"connect", "output", "dump"}:
            continue
        if words and words[0].lower() == "spacegroup":
            skip_next = True
            continue
        kept.append(line)
    parts = ["\n".join(kept).strip(), connections.strip()]
    parts.append(f"output movie cif {stage['prefix']}.cif")
    if needs_restart:
        parts.append(f"dump every 1 {stage['prefix']}.grs")
    return "\n".join(part for part in parts if part)


def _stage_input_from_cif(
    source_cif: Path,
    stage: dict[str, object],
    natural_mult: float,
    library_name: str | None,
    include_periodic_bonds: bool,
    force_no_symmetry: bool = False,
) -> str:
    from pygulp.cluster import prepare_symmetry, render_gulp_input, validate_gin_contract

    atoms = read(source_cif)
    atoms.set_pbc([True, True, True])
    symmetry_mode = "off" if force_no_symmetry else str(stage.get("symmetry_mode", "auto")).lower()
    conventional, asu, metadata = prepare_symmetry(
        atoms,
        symprec=float(stage.get("symprec", 0.05)),
        disabled=symmetry_mode == "off",
    )
    active = symmetry_mode == "auto" and int(metadata.get("symmetry_operations", 1)) > 1
    calculation_atoms = asu if active else conventional
    spacegroup_number = int(metadata.get("gulp_spacegroup_number", 1)) if active else 1

    connections = ""
    if not active:
        tags = infer_molecule_tags_natural_cutoffs(
            conventional,
            include_periodic_bonds=include_periodic_bonds,
            mult=natural_mult,
        )
        conventional.set_tags(tags)
        pairs = infer_natural_cutoff_connections(
            conventional,
            molecule_tag=None,
            same_tag_only=True,
            include_periodic_bonds=include_periodic_bonds,
            index_base=1,
            local_indexing=False,
            mult=natural_mult,
        )
        connections = "\n".join(f"connect {first} {second}" for first, second in pairs)

    stage["n_atoms_asu"] = len(calculation_atoms)
    stage["n_atoms_total"] = len(conventional)
    stage["spacegroup_number"] = spacegroup_number
    options = _stage_options(stage, connections, bool(stage.get("needs_restart")))
    gin = render_gulp_input(
        calculation_atoms,
        str(stage["keywords"]),
        options,
        library_name,
        spacegroup_number=spacegroup_number,
    )
    validate_gin_contract(gin, len(calculation_atoms), spacegroup_number)
    return gin


def write_stage_results(calc_dir: Path, rows: list[dict[str, object]]) -> None:
    (calc_dir / "stage_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (calc_dir / "stages.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STAGE_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STAGE_RESULT_FIELDS})


def execute_stage_plan(plan_path: Path) -> int:
    plan_path = plan_path.expanduser().resolve()
    calc_dir = plan_path.parent
    plan = json.loads(plan_path.read_text())
    stages = plan["stages"]
    managed_heads = _managed_option_heads(stages)
    expected_asu = int(plan["n_atoms_asu"])
    expected_total = int(plan["n_atoms_conventional"])
    validate_atom_counts = bool(plan.get("validate_atom_counts", False))
    expected_structure = plan.get("expected_structure")
    natural_mult = float(plan.get("natural_mult", 1.1))
    if not isinstance(expected_structure, dict):
        expected_structure = None
    results: list[dict[str, object]] = []
    last_cif: Path | None = None

    for legacy_name in ("ginput1.got", "relaxed.cif", "opt_step"):
        legacy_path = calc_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    for index, stage in enumerate(stages):
        prefix = str(stage["prefix"])
        gin_path = calc_dir / f"{prefix}.gin"
        got_path = calc_dir / f"{prefix}.got"
        cif_path = calc_dir / f"{prefix}.cif"
        restart_path = calc_dir / f"{prefix}.grs"
        row: dict[str, object] = {
            "stage": index + 1,
            "name": stage["name"],
            "status": "started",
            "gin": gin_path.name,
            "got": got_path.name,
            "cif": cif_path.name,
            "restart": restart_path.name if stage.get("needs_restart") else "",
            "symmetry_mode": stage.get("symmetry_mode", "auto"),
            "spacegroup_number": stage.get("spacegroup_number", plan.get("spacegroup_number", 1)),
            "message": "",
        }
        results.append(row)
        write_stage_results(calc_dir, results)

        try:
            for stale_path in (got_path, cif_path, restart_path):
                if stale_path.exists():
                    stale_path.unlink()
            if index:
                previous_stage = stages[index - 1]
                if "symmetry_mode" in stage and stage.get("mode") != "rigid_gfnff_symmetry":
                    previous_cif = calc_dir / f"{previous_stage['prefix']}.cif"
                    if not previous_cif.is_file():
                        raise FileNotFoundError(f"Previous stage CIF is missing: {previous_cif.name}")
                    gin_path.write_text(
                        _stage_input_from_cif(
                            source_cif=previous_cif,
                            stage=stage,
                            natural_mult=natural_mult,
                            library_name=str(plan.get("library_name")) if plan.get("library_name") else None,
                            include_periodic_bonds=bool(plan.get("include_periodic_bonds", True)),
                            force_no_symmetry=bool(plan.get("force_no_symmetry", False)),
                        )
                    )
                elif previous_stage.get("mode") == "rigid_gfnff_symmetry":
                    previous_row = results[index - 1]
                    previous_asu = calc_dir / f"{previous_stage['prefix']}_asymmetric_unit.cif"
                    if not previous_asu.is_file():
                        raise FileNotFoundError(
                            f"Previous rigid stage did not create its ASU file: {previous_asu.name}"
                        )
                    from pygulp.rigid import build_symmetric_stage_input

                    gin_path.write_text(
                        build_symmetric_stage_input(
                            calc_dir=calc_dir,
                            source_prefix=str(previous_stage["prefix"]),
                            stage=stage,
                            spacegroup_number=int(previous_row["rigid_spacegroup_number"]),
                            library_name=str(plan.get("library_name")) if plan.get("library_name") else None,
                        )
                    )
                else:
                    previous_restart = calc_dir / f"{previous_stage['prefix']}.grs"
                    if not previous_restart.exists():
                        raise FileNotFoundError(f"Previous stage restart is missing: {previous_restart.name}")
                    gin_path.write_text(
                        rewrite_restart(
                            previous_restart.read_text(),
                            stage,
                            managed_heads,
                            library_name=str(plan.get("library_name")) if plan.get("library_name") else None,
                        )
                    )

            if stage.get("mode") == "rigid_gfnff_symmetry":
                from pygulp.rigid import run_rigid_gfnff_symmetry_stage

                data = run_rigid_gfnff_symmetry_stage(
                    calc_dir=calc_dir,
                    stage=stage,
                    gulp_command=str(plan["gulp_command"]),
                    spacegroup_number=int(plan["spacegroup_number"]),
                )
                row.update({key: data.get(key) for key in STAGE_RESULT_FIELDS if key in data})
                row["rigid_scope"] = data.get("rigid_scope")
                row["rigid_steps_completed"] = data.get("rigid_steps_completed")
                row["rigid_resymmetrized"] = data.get("rigid_resymmetrized", True)
                row["rigid_spacegroup_number"] = data.get("rigid_spacegroup_number")
                row["resymmetrization_status"] = data.get("resymmetrization_status")
                converged = bool(data.get("converged"))
                row["converged"] = converged
                require_convergence = bool(stage.get("require_convergence", False))
                if require_convergence and not converged:
                    raise RuntimeError("Rigid GFNFF stage did not reach force/torque/energy tolerance")
            else:
                command = str(plan["gulp_command"]).replace("PREFIX", prefix)
                completed = subprocess.run(command, cwd=calc_dir, shell=True, check=False)
                data = parse_got(got_path)
                row.update({key: data.get(key) for key in STAGE_RESULT_FIELDS if key in data})
                row["symmetry_mode"] = stage.get("symmetry_mode", "auto")
                row["spacegroup_number"] = stage.get("spacegroup_number", plan.get("spacegroup_number", 1))
                if validate_atom_counts:
                    validate_got_contract(
                        data,
                        int(stage.get("n_atoms_asu") or expected_asu),
                        int(stage.get("n_atoms_total") or expected_total),
                    )
                require_convergence = bool(stage.get("require_convergence", True))
                restart_available = restart_path.is_file()
                recoverable_optimizer_stop = (
                    data.get("gulp_status") == "too_many_failed_attempts"
                    and restart_available
                    and not require_convergence
                )
                if completed.returncode != 0 and not recoverable_optimizer_stop:
                    raise RuntimeError(f"GULP command returned exit code {completed.returncode}")
                if require_convergence and not data.get("completed_normally"):
                    raise RuntimeError("GULP did not report normal completion")
                converged = not bool(stage["is_optimisation"]) or data.get("gulp_status") == "optimisation_achieved"
                row["converged"] = converged
                if bool(stage["is_optimisation"]) and require_convergence and not converged:
                    raise RuntimeError(str(data.get("gulp_status") or "optimisation did not converge"))
            if expected_structure is not None:
                validation = validate_structure_output(
                    cif_path,
                    expected_structure,
                    natural_mult,
                    calc_dir / f"{prefix}.structure_validation.json",
                )
                row["structure_validation"] = "valid"
                row["n_molecules"] = validation["n_molecules"]
                row["n_isolated_atoms"] = validation["n_isolated_atoms"]
            if stage.get("needs_restart") and not restart_path.exists():
                raise FileNotFoundError(f"GULP did not write restart file {restart_path.name}")

            row["status"] = "success" if converged else "completed_nonconverged"
            if not converged:
                row["message"] = f"accepted without convergence: {data.get('gulp_status') or 'not reported'}"
            if cif_path.exists():
                last_cif = cif_path
            write_stage_results(calc_dir, results)
        except Exception as exc:
            row["status"] = "failed"
            row["message"] = str(exc)
            write_stage_results(calc_dir, results)
            return 2

    final_got = calc_dir / f"{stages[-1]['prefix']}.got"
    if final_got.exists():
        shutil.copy2(final_got, calc_dir / "ginput1.got")
    if last_cif is not None:
        shutil.copy2(last_cif, calc_dir / "relaxed.cif")
    final_restart = calc_dir / f"{stages[-1]['prefix']}.grs"
    if final_restart.exists():
        shutil.copy2(final_restart, calc_dir / "opt_step")
    return 0
