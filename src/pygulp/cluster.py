from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np
from ase import Atoms
from ase.io import read, write

from pygulp.molecule.connections import (
    infer_molecule_tags_natural_cutoffs,
    infer_natural_cutoff_connections,
    write_connections,
)
from pygulp.stages import (
    STAGE_RESULT_FIELDS,
    StageSpec,
    execute_stage_plan,
    load_stage_specs,
    parse_got,
    validate_got_contract,
)


DEFAULT_PATTERNS = ("POSCAR*", "*POSCAR*", "*.vasp", "*.poscar", "*.cif", "*.CIF")
SUMMARY_FIELDS = (
    "ID",
    "name",
    "status",
    "formula",
    "n_atoms_input",
    "n_atoms_conventional",
    "n_atoms_asu",
    "n_molecules_final",
    "input_spacegroup",
    "input_spacegroup_number",
    "gulp_spacegroup",
    "gulp_spacegroup_number",
    "final_spacegroup",
    "final_spacegroup_number",
    "symmetry_operations",
    "symmetry_fallback",
    "energy_initial_ev",
    "energy_final_ev",
    "volume",
    "runtime_seconds",
    "density_g_cm3",
    "energy_initial_ev_per_atom",
    "energy_final_ev_per_atom",
    "cif_file",
    "cif_status",
    "completed_stages",
    "total_stages",
    "failed_stage",
)
RELAXED_CIF_SUMMARY_FIELDS = (
    "calculation",
    "status",
    "source_cif",
    "output_cif",
    "spacegroup_symbol",
    "spacegroup_number",
    "n_sites_input",
    "n_sites_output",
    "mode",
    "message",
)
@dataclass
class PreparedJob:
    index: int
    poscar: Path
    work_dir: Path
    calc_dir: Path
    gin_path: Path
    got_path: Path
    input_cif_path: Path
    relaxed_cif_path: Path
    job_script_path: Path | None
    row: dict[str, object]
    job_id: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and submit symmetry-aware GULP SLURM jobs from CIF/POSCAR files."
    )
    parser.add_argument("input_dir", type=Path, nargs="?", help="Directory containing CIF/POSCAR files.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("pygulp_runs"), help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories recursively.")
    parser.add_argument("--pattern", action="append", default=None, help="Glob pattern. Can be passed multiple times.")
    parser.add_argument("--format", default="auto", help="ASE input format override (default: detect CIF/VASP).")
    parser.add_argument("--task-id", type=int, default=None, help="Process only this persistent structure ID.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files.")

    parser.add_argument("--library", default=None, help="Force-field library file stored next to the binary, e.g. reaxff_general.lib.")
    parser.add_argument("--keywords-file", default="keyword.in", help="File with GULP keywords.")
    parser.add_argument("--options-file", default="options.in", help="File with GULP options placed after coordinates.")
    parser.add_argument("--stages-file", type=Path, default=None, help="YAML file describing sequential GULP stages.")
    parser.add_argument("--force", action="store_true", help="Allow existing calculation results to be overwritten.")
    parser.add_argument("--execute-stage-plan", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--prepare-only", action="store_true", help="Only generate CalcFold inputs and job scripts.")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild summary tables from existing work directories.")
    parser.add_argument(
        "--collect-relaxed-cifs",
        action="store_true",
        help="Collect CalcFold/relaxed.cif files into one folder after rewriting them with pymatgen symmetry.",
    )
    parser.add_argument(
        "--relaxed-cif-dir",
        type=Path,
        default=None,
        help="Output directory for collected CIF files. Relative paths are resolved under --output-dir.",
    )
    parser.add_argument(
        "--relaxed-cif-mode",
        "--cif-structure-mode",
        dest="relaxed_cif_mode",
        choices=("refined", "conventional", "primitive"),
        default="refined",
        help="Structure representation used before writing collected CIF files.",
    )
    parser.add_argument(
        "--symprec",
        "--relaxed-cif-symprec",
        dest="symprec",
        type=float,
        default=0.05,
        help="Symmetry precision for input standardisation and final CIF analysis.",
    )
    parser.add_argument(
        "--relaxed-cif-angle-tolerance",
        "--angle-tolerance",
        dest="relaxed_cif_angle_tolerance",
        type=float,
        default=5.0,
        help="Angle tolerance passed to pymatgen when rewriting collected CIF files.",
    )
    parser.add_argument(
        "--relaxed-cif-significant-figures",
        type=int,
        default=8,
        help="Number of significant figures used when writing collected CIF files.",
    )
    parser.add_argument("--submit-jobs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=32,
        help="Maximum number of simultaneously submitted SLURM jobs. Use 0 to submit all at once.",
    )
    parser.add_argument("--poll-interval", type=float, default=15.0, help="Seconds between SLURM status polls.")

    parser.add_argument("--sbatch-command", default="sbatch", help=argparse.SUPPRESS)
    parser.add_argument("--squeue-command", default="squeue", help=argparse.SUPPRESS)
    parser.add_argument("--sacct-command", default="sacct", help=argparse.SUPPRESS)
    parser.add_argument("--gulp-exe", default="gulp", help="GULP executable invoked inside the job script.")
    parser.add_argument("--gulp-command", default=None, help="Full GULP run command. PREFIX is replaced with ginput1.")
    parser.add_argument("--job-script-name", default="job.sh", help="Per-CalcFold job script filename.")
    parser.add_argument(
        "--job-template",
        default="job.sh",
        help="Optional SLURM job template. If the file is missing, a built-in template is used.",
    )
    parser.add_argument("--job-time", default="02:00:00", help="SBATCH walltime for generated job scripts.")
    parser.add_argument("--job-cpus", type=int, default=1, help="SBATCH cpus-per-task for generated job scripts.")
    parser.add_argument("--job-partition", default=None, help="Optional SBATCH partition for generated job scripts.")
    parser.add_argument("--job-account", default=None, help="Optional SBATCH account for generated job scripts.")
    parser.add_argument("--job-module", action="append", default=None, help="Module to load in generated job scripts.")
    parser.add_argument("--job-extra-line", action="append", default=None, help="Extra shell line for generated job scripts.")

    parser.add_argument("--no-connections", action="store_true", help="Do not include generated connect records in GULP input.")
    parser.add_argument("--no-symmetry", action="store_true", help="Write the full input structure as P1.")
    parser.add_argument("--exclude-periodic-bonds", action="store_true", help="Drop bonds through periodic images.")
    parser.add_argument("--natural-mult", type=float, default=1.1, help="ASE natural cutoff multiplier.")
    return parser


def runtime_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(sys.argv[0]).resolve().parent


def log_message(log_path: Path, message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fd:
        fd.write(stamped + "\n")


def collect_structures(
    input_dir: Path,
    recursive: bool,
    patterns: tuple[str, ...],
    excluded_dir: Path | None = None,
) -> list[Path]:
    paths: set[Path] = set()
    excluded_dir = excluded_dir.resolve() if excluded_dir is not None else None
    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in iterator:
            resolved = path.resolve()
            if path.is_file() and (excluded_dir is None or not resolved.is_relative_to(excluded_dir)):
                paths.add(resolved)
    return sorted(paths)


def detect_input_format(path: Path, requested_format: str) -> str | None:
    if requested_format.lower() != "auto":
        return requested_format
    if path.suffix.lower() == ".cif":
        return "cif"
    if "poscar" in path.name.lower() or path.suffix.lower() in {".vasp", ".poscar"}:
        return "vasp"
    return None


def read_structure(path: Path, requested_format: str):
    input_format = detect_input_format(path, requested_format)
    return read(path, format=input_format) if input_format else read(path)


def assign_structure_ids(output_dir: Path, structures: list[Path]) -> list[tuple[int, Path]]:
    manifest_path = output_dir / "structure_manifest.json"
    manifest: dict[str, int] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        manifest = {str(path): int(identifier) for path, identifier in payload.get("structures", {}).items()}

    next_id = max(manifest.values(), default=0) + 1
    for path in structures:
        key = str(path.resolve())
        if key not in manifest:
            manifest[key] = next_id
            next_id += 1

    manifest_text = json.dumps({"structures": manifest}, indent=2, sort_keys=True) + "\n"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(manifest_text)
    temporary_manifest.replace(manifest_path)
    return sorted(((manifest[str(path.resolve())], path) for path in structures), key=lambda item: item[0])


def _periodic_distance(first: np.ndarray, second: np.ndarray, lattice: np.ndarray) -> float:
    delta = np.asarray(first) - np.asarray(second)
    delta -= np.round(delta)
    return float(np.linalg.norm(delta @ lattice))


def _expanded_asu_matches(
    asu: Atoms,
    conventional: Atoms,
    rotations: np.ndarray,
    translations: np.ndarray,
    tolerance: float,
) -> bool:
    # ponytail: quadratic matching is fine for molecular cells; use a spatial index for very large systems.
    expanded: list[tuple[int, np.ndarray]] = []
    lattice = np.asarray(conventional.cell)
    for number, position in zip(asu.get_atomic_numbers(), asu.get_scaled_positions(wrap=True)):
        for rotation, translation in zip(rotations, translations):
            transformed = np.mod(rotation @ position + translation, 1.0)
            if not any(
                number == old_number and _periodic_distance(transformed, old_position, lattice) <= tolerance
                for old_number, old_position in expanded
            ):
                expanded.append((int(number), transformed))

    if len(expanded) != len(conventional):
        return False

    unused = set(range(len(expanded)))
    for number, position in zip(conventional.get_atomic_numbers(), conventional.get_scaled_positions(wrap=True)):
        matches = [
            index
            for index in unused
            if expanded[index][0] == int(number)
            and _periodic_distance(position, expanded[index][1], lattice) <= tolerance
        ]
        if not matches:
            return False
        unused.remove(matches[0])
    return not unused


def prepare_symmetry(atoms: Atoms, symprec: float, disabled: bool = False) -> tuple[Atoms, Atoms, dict[str, object]]:
    atoms = atoms.copy()
    atoms.set_pbc([True, True, True])
    metadata: dict[str, object] = {
        "input_spacegroup": "P1",
        "input_spacegroup_number": 1,
        "gulp_spacegroup": "P1",
        "gulp_spacegroup_number": 1,
        "symmetry_operations": 1,
        "symmetry_fallback": bool(disabled),
    }

    if disabled:
        return atoms, atoms.copy(), metadata

    try:
        import spglib

        cell = (
            np.asarray(atoms.cell),
            atoms.get_scaled_positions(wrap=True),
            atoms.get_atomic_numbers(),
        )
        input_dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if input_dataset is None:
            raise ValueError("spglib could not determine input symmetry")
        metadata["input_spacegroup"] = str(input_dataset.international)
        metadata["input_spacegroup_number"] = int(input_dataset.number)

        standardized = spglib.standardize_cell(
            cell,
            to_primitive=False,
            no_idealize=False,
            symprec=symprec,
        )
        if standardized is None:
            raise ValueError("spglib could not build a conventional cell")
        lattice, positions, numbers = standardized
        conventional = Atoms(numbers=numbers, cell=lattice, scaled_positions=positions, pbc=True)

        dataset = spglib.get_symmetry_dataset(
            (lattice, positions, numbers),
            symprec=symprec,
        )
        if dataset is None:
            raise ValueError("spglib could not determine conventional-cell symmetry")

        representatives: list[int] = []
        seen: set[int] = set()
        for index, equivalent in enumerate(dataset.equivalent_atoms):
            equivalent = int(equivalent)
            if equivalent not in seen:
                seen.add(equivalent)
                representatives.append(index)
        asu = conventional[representatives]
        asu.set_cell(conventional.cell)
        asu.set_pbc(conventional.pbc)

        tolerance = max(float(symprec), 1e-5)
        if not _expanded_asu_matches(
            asu,
            conventional,
            np.asarray(dataset.rotations),
            np.asarray(dataset.translations),
            tolerance,
        ):
            raise ValueError("ASU expansion does not reproduce the conventional cell")

        metadata.update(
            {
                "gulp_spacegroup": str(dataset.international),
                "gulp_spacegroup_number": int(dataset.number),
                "symmetry_operations": int(len(dataset.rotations)),
                "symmetry_fallback": False,
            }
        )
        if int(dataset.number) == 1 and len(asu) != len(conventional):
            raise ValueError("P1 input must contain the complete conventional structure")
        return conventional, asu, metadata
    except Exception as exc:
        metadata["symmetry_fallback"] = True
        metadata["_symmetry_error"] = repr(exc)
        return atoms, atoms.copy(), metadata


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "structure"


def read_required_text_file(path: str, label: str) -> str:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"{label} file not found: {file_path}")
    return file_path.read_text().rstrip()


def detect_forcefield(library_name: str) -> str:
    lowered = library_name.lower()
    if "dreiding" in lowered:
        return "dreiding"
    if "gfnff" in lowered:
        return "gfnff"
    if "reaxff" in lowered:
        return "reaxff"
    raise ValueError(
        f"Could not determine force field from library name '{library_name}'. "
        "Supported names must contain 'reaxff' or 'gfnff'."
    )


def resolve_library_source(library: str, runtime_dir: Path) -> tuple[Path, str]:
    candidate = Path(library).expanduser()
    if candidate.is_absolute():
        library_path = candidate.resolve()
    else:
        search_roots = [runtime_dir]
        project_root = Path(__file__).resolve().parents[2]
        if project_root not in search_roots:
            search_roots.append(project_root)

        library_path = None
        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.exists() and resolved.is_file():
                library_path = resolved
                break
        if library_path is None:
            library_path = (runtime_dir / candidate).resolve()

    if not library_path.exists() or not library_path.is_file():
        raise FileNotFoundError(f"Library file not found next to the binary: {library_path}")

    forcefield = detect_forcefield(library_path.name)
    if forcefield == "dreiding":
        raise ValueError("dreiding libraries are not supported by this build.")
    return library_path, forcefield


def build_options(
    connections_text: str,
    extra_options: str,
    output_cif: str = "relaxed.cif",
    restart_file: str | None = None,
) -> str:
    lines: list[str] = []
    if extra_options:
        lines.append(extra_options.rstrip())
    if connections_text:
        lines.append(connections_text.rstrip())
    lines.append(f"output movie cif {output_cif}")
    if restart_file:
        lines.append(f"dump {restart_file}")
    return "\n".join(lines).rstrip() + "\n"


def molecule_summary(atoms, tags: np.ndarray, local_connections_by_tag: dict[int, list[tuple[int, int]]]) -> list[dict]:
    summary = []
    for tag in sorted(set(tags.tolist())):
        indices = np.where(tags == tag)[0]
        formula = dict(Counter(atoms[int(index)].symbol for index in indices))
        summary.append(
            {
                "tag": int(tag),
                "size": int(len(indices)),
                "formula": formula,
                "indices_1based": [int(index + 1) for index in indices],
                "n_connections": len(local_connections_by_tag.get(int(tag), [])),
            }
        )
    return summary


def render_gulp_input(
    atoms,
    keywords: str,
    options: str,
    library_name: str | None = None,
    spacegroup_number: int = 1,
) -> str:
    lines = [keywords.rstrip(), "title", "ASE calculation", "end", ""]

    if all(atoms.pbc):
        cell_params = atoms.cell.cellpar()
        lines.append("cell")
        lines.append("{:9.6f} {:9.6f} {:9.6f} {:8.5f} {:8.5f} {:8.5f}".format(*cell_params))
        lines.append("frac")
        coords = atoms.get_scaled_positions()
    else:
        lines.append("cart")
        coords = atoms.get_positions()

    charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    if charges.shape[0] != len(atoms):
        charges = np.zeros(len(atoms), dtype=float)

    for symbol, xyz, charge in zip(atoms.get_chemical_symbols(), coords, charges):
        lines.append(f" {symbol:<2} {xyz[0]:10.7f}  {xyz[1]:10.7f}  {xyz[2]:10.7f}  {charge:10.5f}")

    lines.append("")
    lines.extend(["spacegroup", str(int(spacegroup_number)), ""])
    if library_name:
        lines.append(f"library {library_name}")
    lines.append(options.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def validate_gin_contract(gin_text: str, expected_asu: int, expected_spacegroup: int) -> None:
    lines = gin_text.splitlines()
    coordinate_start = next(
        (index + 1 for index, line in enumerate(lines) if line.strip().lower() in {"frac", "cart"}),
        None,
    )
    if coordinate_start is None:
        raise ValueError("Generated GULP input has no frac/cart coordinate block")
    spacegroup_index = next(
        (index for index, line in enumerate(lines[coordinate_start:], coordinate_start) if line.strip().lower() == "spacegroup"),
        None,
    )
    if spacegroup_index is None:
        raise ValueError("Generated GULP input has no spacegroup block")
    coordinate_count = sum(bool(line.strip()) for line in lines[coordinate_start:spacegroup_index])
    spacegroup_value = next((line.strip() for line in lines[spacegroup_index + 1 :] if line.strip()), None)
    if coordinate_count != expected_asu or spacegroup_value != str(expected_spacegroup):
        raise ValueError(
            "Generated GULP input violates symmetry contract: "
            f"coordinates={coordinate_count} (expected {expected_asu}), "
            f"spacegroup={spacegroup_value} (expected {expected_spacegroup})"
        )


def gulp_run_line(args, prefix: str = "ginput1") -> str:
    if args.gulp_command:
        return args.gulp_command.replace("PREFIX", prefix)
    return f'"{args.gulp_exe}" < {prefix}.gin > {prefix}.got'


def stage_gulp_command(args) -> str:
    if args.gulp_command:
        if "PREFIX" not in args.gulp_command:
            raise ValueError("--gulp-command must contain PREFIX in multi-stage mode")
        return args.gulp_command

    template_path = Path(args.job_template).expanduser()
    if template_path.exists():
        for line in template_path.read_text().splitlines():
            if re.search(r"\bgulp\b.*<.*ginput1\.gin.*>.*ginput1\.got", line) and not line.lstrip().startswith("#"):
                return line.strip().replace("ginput1", "PREFIX")
    return f'"{args.gulp_exe}" < PREFIX.gin > PREFIX.got'


def render_job_script(job_name: str, calc_dir: Path, args, run_line: str | None = None) -> str:
    has_custom_run_line = run_line is not None
    run_line = run_line or gulp_run_line(args)
    context = {
        "job_name": job_name[:80],
        "calc_dir": str(calc_dir),
        "gulp_exe": args.gulp_exe,
        "gin": "ginput1.gin",
        "got": "ginput1.got",
        "gulp_run_line": run_line,
    }

    template_path = Path(args.job_template).expanduser()
    if template_path.exists():
        template_text = template_path.read_text()
        rendered = Template(template_text).safe_substitute(context)
        lines = []
        replaced_job_name = False
        replaced_run_line = not has_custom_run_line or "$gulp_run_line" in template_text or "${gulp_run_line}" in template_text
        for line in rendered.splitlines():
            if re.match(r"^\s*#SBATCH\s+(--job-name(?:=|\s+)|-J\s+)", line):
                lines.append(f"#SBATCH --job-name={context['job_name']}")
                replaced_job_name = True
            elif has_custom_run_line and re.search(r"\bgulp\b.*<.*\.gin.*>.*\.got", line):
                lines.append(run_line)
                replaced_run_line = True
            else:
                lines.append(line)
        if not replaced_job_name:
            insert_at = 1 if lines and lines[0].startswith("#!") else 0
            lines.insert(insert_at, f"#SBATCH --job-name={context['job_name']}")
        if not replaced_run_line:
            lines.append(run_line)
        return "\n".join(lines).rstrip() + "\n"

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={context['job_name']}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.job_cpus}",
        f"#SBATCH --time={args.job_time}",
        "#SBATCH --output=slurm_%j.out",
        "#SBATCH --error=slurm_%j.err",
    ]
    if args.job_partition:
        sbatch_lines.append(f"#SBATCH --partition={args.job_partition}")
    if args.job_account:
        sbatch_lines.append(f"#SBATCH --account={args.job_account}")

    body = ["", "set -euo pipefail"]
    for module in args.job_module or []:
        body.append(f"module load {module}")
    for line in args.job_extra_line or []:
        body.append(line)
    body.extend([f'cd "{calc_dir}"', context["gulp_run_line"]])
    return "\n".join(sbatch_lines + body) + "\n"


def stage_runner_line(plan_path: Path) -> str:
    if getattr(sys, "frozen", False):
        command = [str(Path(sys.executable).resolve())]
    else:
        launcher = Path(sys.argv[0]).resolve()
        if launcher.exists() and launcher.suffix == ".py":
            command = [sys.executable, str(launcher)]
        else:
            command = [sys.executable, "-m", "pygulp.cli"]
    command.extend(["--execute-stage-plan", plan_path.name])
    return " ".join(shlex.quote(item) for item in command)


def build_stage_plan(
    stages: list[StageSpec],
    asu_atoms,
    connections_text: str,
    library_name: str | None,
    spacegroup_number: int,
    n_atoms_conventional: int,
    calc_dir: Path,
    args,
) -> tuple[Path, Path]:
    stage_payloads: list[dict[str, object]] = []
    for index, stage in enumerate(stages):
        options_parts = [part for part in (stage.options.rstrip(), connections_text.rstrip()) if part]
        stage_options = "\n".join(options_parts)
        stage_payloads.append(
            {
                "name": stage.name,
                "prefix": stage.prefix,
                "keywords": stage.keywords,
                "options": stage_options,
                "is_optimisation": stage.is_optimisation,
                "require_convergence": stage.require_convergence,
                "needs_restart": index < len(stages) - 1,
            }
        )

    first = stage_payloads[0]
    first_options = build_options(
        connections_text=connections_text,
        extra_options=stages[0].options,
        output_cif=f"{first['prefix']}.cif",
        restart_file=f"{first['prefix']}.grs" if first["needs_restart"] else None,
    )
    first_gin = render_gulp_input(
        asu_atoms,
        stages[0].keywords,
        first_options,
        library_name,
        spacegroup_number=spacegroup_number,
    )
    validate_gin_contract(first_gin, len(asu_atoms), spacegroup_number)
    first_gin_path = calc_dir / f"{first['prefix']}.gin"
    first_gin_path.write_text(first_gin)
    (calc_dir / "ginput1.gin").write_text(first_gin)

    plan = {
        "n_atoms_asu": len(asu_atoms),
        "n_atoms_conventional": n_atoms_conventional,
        "spacegroup_number": spacegroup_number,
        "gulp_command": stage_gulp_command(args),
        "stages": stage_payloads,
    }
    plan_path = calc_dir / "stage_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path, first_gin_path


def apply_stage_results(row: dict[str, object], calc_dir: Path) -> bool:
    result_path = calc_dir / "stage_results.json"
    if not result_path.exists():
        return False
    results = json.loads(result_path.read_text())
    plan_path = calc_dir / "stage_plan.json"
    if plan_path.exists():
        row["total_stages"] = len(json.loads(plan_path.read_text()).get("stages", results))
    else:
        row["total_stages"] = len(results)
    row["completed_stages"] = sum(
        item.get("status") in {"success", "completed_nonconverged"} for item in results
    )
    failed = next((item for item in results if item.get("status") == "failed"), None)
    row["failed_stage"] = failed.get("name") if failed else None
    if failed:
        row["status"] = f"stage_failed:{failed.get('name')}"
    elif (
        results
        and row["completed_stages"] == row["total_stages"]
        and row.get("status") in {"started", "prepared", "submitted", "missing", "unknown", "completed_without_output"}
    ):
        row["status"] = "success"
    return True


def write_global_stage_summary(output_dir: Path) -> Path | None:
    rows: list[dict[str, object]] = []
    for work_dir in sorted(output_dir.glob("[0-9]*_*")):
        result_path = work_dir / "CalcFold" / "stage_results.json"
        if not result_path.exists():
            continue
        identifier_match = re.match(r"^(\d+)_", work_dir.name)
        identifier = int(identifier_match.group(1)) if identifier_match else work_dir.name
        for stage in json.loads(result_path.read_text()):
            rows.append({"ID": identifier, "calculation": work_dir.name, **stage})
    if not rows:
        return None
    fields = ("ID", "calculation", *STAGE_RESULT_FIELDS)
    output_path = output_dir / "stages.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output_path


def write_summary(output_dir: Path, rows: list[dict[str, object]], basename: str = "summary") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}.csv"
    jsonl_path = output_dir / f"{basename}.jsonl"

    with csv_path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    with jsonl_path.open("w") as fd:
        for row in rows:
            payload = {field: row.get(field) for field in SUMMARY_FIELDS}
            fd.write(json.dumps(payload) + "\n")

    try:
        import pandas as pd

        pd.DataFrame(
            [{field: row.get(field) for field in SUMMARY_FIELDS} for row in rows],
            columns=SUMMARY_FIELDS,
        ).to_excel(output_dir / f"{basename}.xlsx", index=False, sheet_name="Summary")
    except ImportError as exc:
        raise RuntimeError("Writing XLSX requires pandas and openpyxl") from exc

    return csv_path, jsonl_path


def write_relaxed_cif_summary(output_dir: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "relaxed_cifs_summary.csv"
    jsonl_path = output_dir / "relaxed_cifs_summary.jsonl"

    with csv_path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=RELAXED_CIF_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RELAXED_CIF_SUMMARY_FIELDS})

    with jsonl_path.open("w") as fd:
        for row in rows:
            payload = {field: row.get(field) for field in RELAXED_CIF_SUMMARY_FIELDS}
            fd.write(json.dumps(payload) + "\n")

    return csv_path, jsonl_path


def resolve_relaxed_cif_output_dir(output_dir: Path, relaxed_cif_dir: Path | None) -> Path:
    if relaxed_cif_dir is None:
        return output_dir / "relaxed_cifs"

    resolved = relaxed_cif_dir.expanduser()
    if not resolved.is_absolute():
        resolved = output_dir / resolved
    return resolved.resolve()


def structure_for_relaxed_cif_mode(analyzer, mode: str):
    if mode == "refined":
        return analyzer.get_refined_structure()
    if mode == "conventional":
        return analyzer.get_conventional_standard_structure()
    if mode == "primitive":
        structure = analyzer.get_primitive_standard_structure()
        if structure is None:
            structure = analyzer.find_primitive()
        if structure is None:
            raise ValueError("pymatgen could not build a primitive structure")
        return structure
    raise ValueError(f"Unknown relaxed CIF mode: {mode}")


def rewrite_relaxed_cif_with_symmetry(
    source_cif: Path,
    output_cif: Path,
    mode: str,
    symprec: float,
    angle_tolerance: float,
    significant_figures: int,
) -> dict[str, object]:
    row: dict[str, object] = {field: None for field in RELAXED_CIF_SUMMARY_FIELDS}
    row.update(
        {
            "status": "started",
            "source_cif": str(source_cif),
            "output_cif": str(output_cif),
            "mode": mode,
        }
    )
    structure = None
    captured_warnings: list[str] = []

    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structure = Structure.from_file(str(source_cif))
            row["n_sites_input"] = len(structure)

            analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            spacegroup_symbol = analyzer.get_space_group_symbol()
            spacegroup_number = analyzer.get_space_group_number()
            converted = structure_for_relaxed_cif_mode(analyzer, mode)

            output_cif.parent.mkdir(parents=True, exist_ok=True)
            writer = CifWriter(
                converted,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                significant_figures=significant_figures,
                refine_struct=False,
            )
            writer.write_file(str(output_cif))
            Structure.from_file(str(output_cif))
            captured_warnings = [str(item.message) for item in caught]

        row["spacegroup_symbol"] = spacegroup_symbol
        row["spacegroup_number"] = spacegroup_number
        row["n_sites_output"] = len(converted)
        if spacegroup_number == 1:
            row["status"] = "written_p1"
            row["message"] = "symmetry resolved as P1 with the selected tolerances"
        else:
            row["status"] = "written"
            row["message"] = "; ".join(captured_warnings[:3])
    except Exception as exc:
        if structure is None:
            row["status"] = "failed"
            row["message"] = repr(exc)
            return row

        try:
            from pymatgen.io.cif import CifWriter

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output_cif.parent.mkdir(parents=True, exist_ok=True)
                writer = CifWriter(
                    structure,
                    symprec=None,
                    significant_figures=significant_figures,
                )
                writer.write_file(str(output_cif))
                from pymatgen.core import Structure

                Structure.from_file(str(output_cif))
            row["status"] = "written_p1_fallback"
            row["n_sites_output"] = len(structure)
            row["spacegroup_symbol"] = "P 1"
            row["spacegroup_number"] = 1
            row["message"] = f"symmetry conversion failed, wrote input structure without symmetry: {exc!r}"
        except Exception as fallback_exc:
            row["status"] = "failed"
            row["message"] = f"{exc!r}; fallback write failed: {fallback_exc!r}"

    return row


def detect_relaxed_cif_symmetry(
    relaxed_cif_path: Path,
    symprec: float,
    angle_tolerance: float,
) -> tuple[str | None, int | None]:
    if not relaxed_cif_path.exists():
        return None, None

    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(str(relaxed_cif_path))
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            return analyzer.get_space_group_symbol(), analyzer.get_space_group_number()
    except Exception:
        return None, None


def base_row(identifier: int, name: str, source: Path, work_dir: Path) -> dict[str, object]:
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(
        {
            "ID": identifier,
            "name": name,
            "status": "started",
            "_source": str(source),
            "_work_dir": str(work_dir),
        }
    )
    return row


def copy_selected_library(library_source: Path, calc_dir: Path) -> str:
    calc_dir.mkdir(parents=True, exist_ok=True)
    destination = calc_dir / library_source.name
    shutil.copy2(library_source, destination)
    return destination.name


def enrich_row_from_outputs(
    row: dict[str, object],
    got_path: Path,
    relaxed_cif_path: Path | None = None,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> dict[str, object]:
    got_data = parse_got(got_path)
    expected_asu = row.get("n_atoms_asu")
    expected_total = row.get("n_atoms_conventional")
    if isinstance(expected_asu, int) and isinstance(expected_total, int):
        validate_got_contract(got_data, expected_asu, expected_total)
    row["energy_initial_ev"] = got_data["energy_initial_ev"]
    row["energy_final_ev"] = got_data["energy_final_ev"]
    row["volume"] = got_data["volume"]
    row["runtime_seconds"] = got_data["runtime_seconds"]

    n_atoms = row.get("n_atoms_conventional")
    if isinstance(n_atoms, int) and n_atoms > 0:
        if got_data["energy_initial_ev"] is not None:
            row["energy_initial_ev_per_atom"] = float(got_data["energy_initial_ev"]) / n_atoms
        if got_data["energy_final_ev"] is not None:
            row["energy_final_ev_per_atom"] = float(got_data["energy_final_ev"]) / n_atoms

    volume_value = got_data["volume"]
    if volume_value is not None:
        try:
            volume_float = float(volume_value)
        except (TypeError, ValueError):
            volume_float = None
        if volume_float and volume_float > 0:
            total_mass_amu = row.get("_total_mass_amu")
            if isinstance(total_mass_amu, (int, float)) and total_mass_amu > 0:
                row["density_g_cm3"] = float(total_mass_amu) * 1.66053906660 / volume_float

    if relaxed_cif_path is not None:
        symmetry, symmetry_number = detect_relaxed_cif_symmetry(
            relaxed_cif_path=relaxed_cif_path,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        row["final_spacegroup"] = symmetry
        row["final_spacegroup_number"] = symmetry_number

    return got_data


def export_final_cif(row: dict[str, object], args, log_path: Path) -> None:
    identifier = int(row["ID"])
    source_cif = Path(str(row["_relaxed_cif_path"]))
    output_dir = resolve_relaxed_cif_output_dir(args.output_dir, args.relaxed_cif_dir)
    output_cif = output_dir / f"{identifier}.cif"
    row["cif_file"] = output_cif.name

    if not source_cif.exists():
        row["cif_status"] = "missing_relaxed_cif"
        return

    cif_row = rewrite_relaxed_cif_with_symmetry(
        source_cif=source_cif,
        output_cif=output_cif,
        mode=args.relaxed_cif_mode,
        symprec=args.symprec,
        angle_tolerance=args.relaxed_cif_angle_tolerance,
        significant_figures=args.relaxed_cif_significant_figures,
    )
    row["cif_status"] = cif_row["status"]
    row["final_spacegroup"] = cif_row["spacegroup_symbol"]
    row["final_spacegroup_number"] = cif_row["spacegroup_number"]

    if str(cif_row["status"]).startswith("written"):
        try:
            atoms = read(output_cif)
            atoms.set_pbc([True, True, True])
            tags = infer_molecule_tags_natural_cutoffs(
                atoms,
                include_periodic_bonds=not args.exclude_periodic_bonds,
                mult=args.natural_mult,
            )
            row["n_molecules_final"] = len(set(int(tag) for tag in tags))
        except Exception as exc:
            row["cif_status"] = f"{cif_row['status']}; molecule_count_failed"
            log_message(log_path, f"[{identifier}] final molecule count failed: {exc!r}")


def export_final_cifs(rows: list[dict[str, object]], args, log_path: Path) -> None:
    for row in rows:
        export_final_cif(row, args, log_path)


def determine_final_status(slurm_state: str | None, got_data: dict[str, object], relaxed_cif_path: Path) -> str:
    normalized = normalize_slurm_state(slurm_state)
    if normalized and normalized not in {"COMPLETED", "UNKNOWN"}:
        return normalized.lower()

    gulp_status = got_data.get("gulp_status")
    if gulp_status == "optimisation_achieved":
        return "success" if relaxed_cif_path.exists() else "got_without_cif"
    if isinstance(gulp_status, str):
        return gulp_status

    if normalized == "COMPLETED":
        return "completed_without_output"
    return "unknown"


def normalize_slurm_state(state: str | None) -> str | None:
    if not state:
        return None
    return state.split()[0].strip().upper()


def prepare_structure(
    index: int,
    poscar: Path,
    args,
    keywords: str,
    extra_options: str,
    library_source: Path | None,
    log_path: Path,
    stages: list[StageSpec] | None = None,
) -> PreparedJob:
    name = f"{index:05d}_{sanitize_name(poscar.stem)}"
    work_dir = args.output_dir / name
    calc_dir = work_dir / "CalcFold"
    gin_path = calc_dir / "ginput1.gin"
    got_path = calc_dir / "ginput1.got"
    input_cif_path = work_dir / "input.cif"
    relaxed_cif_path = calc_dir / "relaxed.cif"
    row = base_row(index, name, poscar, work_dir)
    row["_relaxed_cif_path"] = str(relaxed_cif_path)

    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        calc_dir.mkdir(parents=True, exist_ok=True)
        existing_results = [
            path
            for path in (got_path, relaxed_cif_path, calc_dir / "stage_results.json")
            if path.exists()
        ]
        if existing_results and not args.force:
            log_message(
                log_path,
                f"[{index}] existing results preserved; use --force to overwrite: "
                + ", ".join(path.name for path in existing_results),
            )
            existing_row = collect_existing_result(index, poscar, args, log_path)
            return PreparedJob(
                index=index,
                poscar=poscar,
                work_dir=work_dir,
                calc_dir=calc_dir,
                gin_path=gin_path,
                got_path=got_path,
                input_cif_path=input_cif_path,
                relaxed_cif_path=relaxed_cif_path,
                job_script_path=None,
                row=existing_row,
            )
        for stale_file in (got_path, relaxed_cif_path, calc_dir / "opt_step"):
            if stale_file.exists() and stale_file.is_file():
                stale_file.unlink()
        for stale_file in (calc_dir / "stage_results.json", calc_dir / "stages.csv"):
            if stale_file.exists():
                stale_file.unlink()

        input_atoms = read_structure(poscar, args.format)
        input_atoms.set_pbc([True, True, True])
        row["n_atoms_input"] = len(input_atoms)
        row["formula"] = input_atoms.get_chemical_formula(empirical=True)

        shutil.copy2(poscar, work_dir / poscar.name)
        write(input_cif_path, input_atoms)

        atoms, asu_atoms, symmetry = prepare_symmetry(
            input_atoms,
            symprec=args.symprec,
            disabled=args.no_symmetry,
        )
        row.update({key: value for key, value in symmetry.items() if not key.startswith("_")})
        row["n_atoms_conventional"] = len(atoms)
        row["n_atoms_asu"] = len(asu_atoms)
        row["_total_mass_amu"] = float(atoms.get_masses().sum())

        write(work_dir / "standardized_full.cif", atoms)
        write(work_dir / "asymmetric_unit.xyz", asu_atoms)
        symmetry_payload = {
            **{key: value for key, value in symmetry.items() if not key.startswith("_")},
            "n_atoms_input": len(input_atoms),
            "n_atoms_conventional": len(atoms),
            "n_atoms_asu": len(asu_atoms),
            "symprec": args.symprec,
        }
        (work_dir / "symmetry.json").write_text(json.dumps(symmetry_payload, indent=2) + "\n")
        if symmetry.get("_symmetry_error"):
            log_message(log_path, f"[{index}] symmetry fallback to P1: {symmetry['_symmetry_error']}")

        tags = infer_molecule_tags_natural_cutoffs(
            atoms,
            include_periodic_bonds=not args.exclude_periodic_bonds,
            mult=args.natural_mult,
        )
        atoms.set_tags(tags)

        local_connections_by_tag: dict[int, list[tuple[int, int]]] = {}
        for tag in sorted(set(tags.tolist())):
            local_connections = infer_natural_cutoff_connections(
                atoms,
                molecule_tag=int(tag),
                same_tag_only=True,
                include_periodic_bonds=not args.exclude_periodic_bonds,
                index_base=1,
                local_indexing=True,
                mult=args.natural_mult,
            )
            local_connections_by_tag[int(tag)] = local_connections
            write_connections(local_connections, work_dir / f"connections_tag_{int(tag)}")

        global_connections = infer_natural_cutoff_connections(
            atoms,
            molecule_tag=None,
            same_tag_only=True,
            include_periodic_bonds=not args.exclude_periodic_bonds,
            index_base=1,
            local_indexing=False,
            mult=args.natural_mult,
        )
        write_connections(global_connections, work_dir / "connections")

        summary = molecule_summary(atoms, tags, local_connections_by_tag)
        (work_dir / "molecule_summary.json").write_text(json.dumps(summary, indent=2))

        symmetry_is_active = int(row["symmetry_operations"] or 1) > 1
        connections_text = ""
        if not args.no_connections and not symmetry_is_active:
            connections_text = (work_dir / "connections").read_text()
        elif symmetry_is_active:
            log_message(log_path, f"[{index}] full-cell connect records omitted for ASU input")
        generated_options = build_options(connections_text=connections_text, extra_options=extra_options)
        (work_dir / "keyword.in").write_text(keywords + "\n")
        (work_dir / "options.in").write_text(extra_options + ("\n" if extra_options else ""))
        (work_dir / "generated_options.in").write_text(generated_options)

        library_name = copy_selected_library(library_source, calc_dir) if library_source is not None else None
        spacegroup_number = int(row["gulp_spacegroup_number"] or 1)
        if stages:
            if args.stages_file is not None:
                shutil.copy2(args.stages_file, work_dir / "stages.yaml")
            plan_path, _ = build_stage_plan(
                stages=stages,
                asu_atoms=asu_atoms,
                connections_text=connections_text,
                library_name=library_name,
                spacegroup_number=spacegroup_number,
                n_atoms_conventional=len(atoms),
                calc_dir=calc_dir,
                args=args,
            )
            run_line = stage_runner_line(plan_path)
            row["total_stages"] = len(stages)
            row["completed_stages"] = 0
        else:
            gin_text = render_gulp_input(
                asu_atoms,
                keywords,
                generated_options,
                library_name,
                spacegroup_number=spacegroup_number,
            )
            validate_gin_contract(gin_text, len(asu_atoms), spacegroup_number)
            gin_path.write_text(gin_text)
            run_line = None

        job_script_path = calc_dir / args.job_script_name
        job_script_path.write_text(render_job_script(name, calc_dir, args, run_line=run_line))
        job_script_path.chmod(0o755)

        row["status"] = "prepared"
        return PreparedJob(
            index=index,
            poscar=poscar,
            work_dir=work_dir,
            calc_dir=calc_dir,
            gin_path=gin_path,
            got_path=got_path,
            input_cif_path=input_cif_path,
            relaxed_cif_path=relaxed_cif_path,
            job_script_path=job_script_path,
            row=row,
        )
    except Exception as exc:
        row["status"] = "failed"
        log_message(log_path, f"[{index}] prepare failed for {poscar}: {exc!r}")
        return PreparedJob(
            index=index,
            poscar=poscar,
            work_dir=work_dir,
            calc_dir=calc_dir,
            gin_path=gin_path,
            got_path=got_path,
            input_cif_path=input_cif_path,
            relaxed_cif_path=relaxed_cif_path,
            job_script_path=None,
            row=row,
        )


def submit_job(job_script: Path, sbatch_command: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            [sbatch_command, job_script.name],
            cwd=str(job_script.parent),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None, f"SBATCH command not found: {sbatch_command}"

    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    return (match.group(1) if match else result.stdout.strip()), None


def query_active_job_ids(job_ids: list[str], squeue_command: str) -> set[str]:
    if not job_ids:
        return set()

    try:
        result = subprocess.run(
            [squeue_command, "-h", "-o", "%A", "-j", ",".join(job_ids)],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"SQUEUE command not found: {squeue_command}") from exc

    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "squeue failed")

    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def query_final_states(job_ids: list[str], sacct_command: str) -> dict[str, str]:
    if not job_ids:
        return {}

    try:
        result = subprocess.run(
            [sacct_command, "-n", "-P", "-o", "JobIDRaw,State", "-j", ",".join(job_ids)],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    if result.returncode != 0:
        return {}

    requested = set(job_ids)
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        raw_job_id, raw_state = line.split("|", 1)
        root_job_id = raw_job_id.split(".", 1)[0]
        if root_job_id not in requested:
            continue
        state = normalize_slurm_state(raw_state)
        if state is None:
            continue
        if root_job_id not in states or raw_job_id == root_job_id:
            states[root_job_id] = state
    return states


def wait_for_jobs(
    prepared_jobs: list[PreparedJob],
    rows: list[dict[str, object]],
    args,
    summary_basename: str,
    log_path: Path,
) -> None:
    pending = deque(job for job in prepared_jobs if job.job_script_path is not None and job.row["status"] == "prepared")
    active: dict[str, PreparedJob] = {}
    max_parallel = len(pending) if args.max_parallel <= 0 else args.max_parallel

    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.popleft()
            assert job.job_script_path is not None
            job_id, submit_error = submit_job(job.job_script_path, args.sbatch_command)
            if submit_error:
                job.row["status"] = "submit_failed"
                log_message(log_path, f"[{job.index}] submit failed for {job.poscar}: {submit_error}")
                write_summary(args.output_dir, rows, summary_basename)
                continue

            job.job_id = job_id
            job.row["status"] = "submitted"
            active[job_id] = job
            log_message(log_path, f"[{job.index}] submitted {job.poscar.name} as job {job_id}")
            write_summary(args.output_dir, rows, summary_basename)

        if not active:
            continue

        time.sleep(max(args.poll_interval, 1.0))
        active_ids = query_active_job_ids(list(active), args.squeue_command)
        finished_ids = [job_id for job_id in list(active) if job_id not in active_ids]
        if not finished_ids:
            continue

        final_states = query_final_states(finished_ids, args.sacct_command)
        for job_id in finished_ids:
            job = active.pop(job_id)
            slurm_state = final_states.get(job_id, "UNKNOWN")
            try:
                has_stage_results = apply_stage_results(job.row, job.calc_dir)
                if not (has_stage_results and job.row.get("failed_stage")):
                    got_data = enrich_row_from_outputs(
                        row=job.row,
                        got_path=job.got_path,
                        relaxed_cif_path=job.relaxed_cif_path,
                        symprec=args.symprec,
                        angle_tolerance=args.relaxed_cif_angle_tolerance,
                    )
                    job.row["status"] = determine_final_status(slurm_state, got_data, job.relaxed_cif_path)
                    if has_stage_results:
                        apply_stage_results(job.row, job.calc_dir)
            except Exception as exc:
                job.row["status"] = "postprocess_failed"
                log_message(log_path, f"[{job.index}] postprocess failed for job {job_id}: {exc!r}")
            log_message(log_path, f"[{job.index}] finished job {job_id} with status {job.row['status']}")
            write_summary(args.output_dir, rows, summary_basename)


def collect_existing_result(index: int, poscar: Path, args, log_path: Path) -> dict[str, object]:
    name = f"{index:05d}_{sanitize_name(poscar.stem)}"
    work_dir = args.output_dir / name
    calc_dir = work_dir / "CalcFold"
    gin_path = calc_dir / "ginput1.gin"
    got_path = calc_dir / "ginput1.got"
    relaxed_cif_path = calc_dir / "relaxed.cif"
    standardized_cif_path = work_dir / "standardized_full.cif"
    symmetry_path = work_dir / "symmetry.json"

    row = base_row(index, name, poscar, work_dir)
    row["_relaxed_cif_path"] = str(relaxed_cif_path)
    row["status"] = "missing"

    try:
        input_atoms = read_structure(poscar, args.format)
        row["n_atoms_input"] = len(input_atoms)
        row["formula"] = input_atoms.get_chemical_formula(empirical=True)
    except Exception:
        pass

    if symmetry_path.exists():
        row.update(json.loads(symmetry_path.read_text()))

    try:
        calculation_atoms = read(standardized_cif_path) if standardized_cif_path.exists() else input_atoms
        row["_total_mass_amu"] = float(calculation_atoms.get_masses().sum())
        row["n_atoms_conventional"] = len(calculation_atoms)
    except Exception:
        pass

    if got_path.exists():
        try:
            got_data = enrich_row_from_outputs(
                row=row,
                got_path=got_path,
                relaxed_cif_path=relaxed_cif_path,
                symprec=args.symprec,
                angle_tolerance=args.relaxed_cif_angle_tolerance,
            )
            row["status"] = determine_final_status("COMPLETED", got_data, relaxed_cif_path)
        except Exception as exc:
            row["status"] = "collect_failed"
            log_message(log_path, f"[{index}] collect failed for {poscar}: {exc!r}")
    elif gin_path.exists():
        row["status"] = "prepared"

    apply_stage_results(row, calc_dir)

    return row


def collect_relaxed_cifs(args, log_path: Path) -> tuple[list[dict[str, object]], Path]:
    relaxed_cif_dir = resolve_relaxed_cif_output_dir(args.output_dir, args.relaxed_cif_dir)
    work_dirs = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_dir() and (path / "CalcFold").is_dir()
    )
    if not work_dirs:
        raise FileNotFoundError(f"No CalcFold directories found in {args.output_dir}")

    relaxed_cif_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for work_dir in work_dirs:
        identifier_match = re.match(r"^(\d+)_", work_dir.name)
        identifier = int(identifier_match.group(1)) if identifier_match else work_dir.name
        source_cif = work_dir / "CalcFold" / "relaxed.cif"
        output_cif = relaxed_cif_dir / f"{identifier}.cif"

        if not source_cif.exists():
            row = {field: None for field in RELAXED_CIF_SUMMARY_FIELDS}
            row.update(
                {
                    "calculation": work_dir.name,
                    "status": "missing_relaxed_cif",
                    "source_cif": str(source_cif),
                    "output_cif": str(output_cif),
                    "mode": args.relaxed_cif_mode,
                    "message": "CalcFold/relaxed.cif does not exist",
                }
            )
            rows.append(row)
            log_message(log_path, f"[{work_dir.name}] missing relaxed.cif")
            continue

        row = rewrite_relaxed_cif_with_symmetry(
            source_cif=source_cif,
            output_cif=output_cif,
            mode=args.relaxed_cif_mode,
            symprec=args.symprec,
            angle_tolerance=args.relaxed_cif_angle_tolerance,
            significant_figures=args.relaxed_cif_significant_figures,
        )
        row["calculation"] = work_dir.name
        rows.append(row)

        sg = row.get("spacegroup_symbol") or "unknown"
        status = row.get("status")
        log_message(log_path, f"[{work_dir.name}] collected relaxed.cif as {output_cif.name} ({status}, {sg})")

    return rows, relaxed_cif_dir


def run_pipeline(args) -> int:
    if args.input_dir is None:
        raise ValueError("input_dir is required unless --execute-stage-plan is used")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.stages_file is not None:
        args.stages_file = args.stages_file.expanduser().resolve()
        if not args.stages_file.is_file():
            raise FileNotFoundError(f"Stages file not found: {args.stages_file}")
    if args.collect_relaxed_cifs and (args.collect_only or args.prepare_only):
        raise ValueError("--collect-relaxed-cifs cannot be combined with --collect-only or --prepare-only.")

    if args.collect_relaxed_cifs:
        if not args.output_dir.is_dir():
            raise NotADirectoryError(f"Output directory does not exist: {args.output_dir}")
        log_path = args.output_dir / "dispatcher.log"
        rows, relaxed_cif_dir = collect_relaxed_cifs(args, log_path)
        csv_path, _ = write_relaxed_cif_summary(args.output_dir, rows)
        written_count = sum(str(row.get("status", "")).startswith("written") for row in rows)
        log_message(
            log_path,
            f"Collected {written_count}/{len(rows)} relaxed CIF files into {relaxed_cif_dir}; wrote summary to {csv_path}",
        )
        return 0

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "dispatcher.log"

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    excluded_dir = args.output_dir if args.output_dir.is_relative_to(args.input_dir) else None
    poscars = collect_structures(args.input_dir, args.recursive, patterns, excluded_dir=excluded_dir)
    if args.limit is not None:
        poscars = poscars[: args.limit]
    if not poscars:
        raise FileNotFoundError(f"No input structures found in {args.input_dir} for patterns {patterns}")

    indexed_poscars = assign_structure_ids(args.output_dir, poscars)
    if args.task_id is not None:
        indexed_poscars = [item for item in indexed_poscars if item[0] == args.task_id]
        if not indexed_poscars:
            raise IndexError(f"No input structure has ID {args.task_id}")

    summary_basename = f"summary_task_{args.task_id:05d}" if args.task_id is not None else "summary"

    if args.collect_only:
        rows = [collect_existing_result(index, poscar, args, log_path) for index, poscar in indexed_poscars]
        export_final_cifs(rows, args, log_path)
        write_global_stage_summary(args.output_dir)
        csv_path, _ = write_summary(args.output_dir, rows, summary_basename)
        log_message(log_path, f"Wrote summary to {csv_path}")
        return 0

    options_path = Path(args.options_file).expanduser()
    if args.stages_file is not None and not options_path.exists():
        extra_options = ""
    else:
        extra_options = read_required_text_file(args.options_file, "options")
    stages = load_stage_specs(args.stages_file, extra_options) if args.stages_file is not None else None
    keywords = stages[0].keywords if stages else read_required_text_file(args.keywords_file, "keywords")
    library_source = None
    if args.library:
        library_source, forcefield = resolve_library_source(args.library, runtime_directory())
        log_message(log_path, f"Using {forcefield} library {library_source.name}")
    else:
        log_message(log_path, "No --library supplied; GULP input will be generated without a library directive.")

    rows: list[dict[str, object]] = []
    prepared_jobs: list[PreparedJob] = []
    for index, poscar in indexed_poscars:
        log_message(log_path, f"[ID {index}] preparing {poscar}")
        prepared = prepare_structure(index, poscar, args, keywords, extra_options, library_source, log_path, stages=stages)
        rows.append(prepared.row)
        if prepared.job_script_path is not None and prepared.row["status"] == "prepared":
            prepared_jobs.append(prepared)
        write_summary(args.output_dir, rows, summary_basename)

    if args.prepare_only:
        csv_path, _ = write_summary(args.output_dir, rows, summary_basename)
        log_message(log_path, f"Wrote summary to {csv_path}")
        return 0

    if prepared_jobs:
        wait_for_jobs(prepared_jobs, rows, args, summary_basename, log_path)
    write_global_stage_summary(args.output_dir)
    export_final_cifs(rows, args, log_path)
    csv_path, _ = write_summary(args.output_dir, rows, summary_basename)
    log_message(log_path, f"Wrote summary to {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute_stage_plan is not None:
        return execute_stage_plan(args.execute_stage_plan)
    return run_pipeline(args)
