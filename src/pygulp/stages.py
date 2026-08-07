from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageSpec:
    name: str
    keywords: str
    options: str
    require_convergence: bool = True
    validate_atom_counts: bool = False

    @property
    def prefix(self) -> str:
        return self.name

    @property
    def is_optimisation(self) -> bool:
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
    "gin",
    "got",
    "cif",
    "restart",
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

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Stage #{index} must be a mapping")
        raw_name = entry.get("name")
        keywords = entry.get("keywords")
        options = entry.get("options", default_options)
        require_convergence = entry.get("require_convergence", True)
        validate_atom_counts = entry.get("validate_atom_counts", default_validate_atom_counts)
        if not isinstance(raw_name, str) or not isinstance(keywords, str) or not keywords.strip():
            raise ValueError(f"Stage #{index} requires non-empty string fields 'name' and 'keywords'")
        if not isinstance(options, str):
            raise ValueError(f"Stage #{index} field 'options' must be a string")
        if not isinstance(require_convergence, bool):
            raise ValueError(f"Stage #{index} field 'require_convergence' must be true or false")
        if not isinstance(validate_atom_counts, bool):
            raise ValueError(f"Stage #{index} field 'validate_atom_counts' must be true or false")
        cleaned_name = sanitize_stage_name(raw_name)
        if cleaned_name in used_names:
            raise ValueError(f"Duplicate stage name: {raw_name}")
        used_names.add(cleaned_name)
        name = f"{index:02d}_{cleaned_name}"
        stages.append(
            StageSpec(
                name=name,
                keywords=keywords.rstrip(),
                options=options.rstrip(),
                require_convergence=require_convergence,
                validate_atom_counts=validate_atom_counts,
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
        "volume": None,
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
    volumes: list[str] = []
    runtime_seconds: float | None = None
    cpu_seconds: float | None = None
    for line in got_path.read_text(errors="replace").splitlines():
        energy_match = re.search(r"Total lattice energy\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*eV", line)
        if energy_match:
            energies.append(float(energy_match.group(1)))
        volume_match = re.search(r"cell volume\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", line, re.I)
        if volume_match:
            volumes.append(volume_match.group(1))
        gnorm_match = re.search(r"Final Gnorm\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", line, re.I)
        if gnorm_match:
            data["gnorm"] = float(gnorm_match.group(1))
        runtime_match = re.search(
            r"Time to end of optimisation\s*=\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*seconds",
            line,
            re.I,
        )
        if runtime_match:
            runtime_seconds = float(runtime_match.group(1))
        cpu_match = re.search(r"Total CPU time\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", line)
        if cpu_match:
            cpu_seconds = float(cpu_match.group(1))
        irreducible_match = re.search(
            r"(?:Number of irreducible atoms/shells|Number of irreducible atoms|Number of atoms/shells in asym(?:ym)?metrical unit)\s*=\s*(\d+)",
            line,
            re.I,
        )
        if irreducible_match:
            data["n_atoms_irreducible"] = int(irreducible_match.group(1))
        total_match = re.search(
            r"(?:^\s*Total number atoms/shells\s*=|^\s*Total number of atoms\s*=|^\s*Number of atoms in unit cell\s*=|^\s*Number of atoms/shells in unit cell\s*=|^\s*Number of atoms\s*=)\s*(\d+)",
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

    if energies:
        data["energy_initial_ev"] = energies[0]
        data["energy_final_ev"] = energies[-1]
    if volumes:
        data["volume"] = volumes[-1]
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


def rewrite_restart(restart_text: str, stage: dict[str, object], managed_heads: set[str]) -> str:
    lines = restart_text.splitlines()
    title_index = next((index for index, line in enumerate(lines) if line.strip().lower() == "title"), None)
    if title_index is None:
        raise ValueError("GULP restart does not contain a title block")

    filtered = []
    for line in lines[title_index:]:
        words = line.split()
        if words and words[0].lower() in managed_heads:
            continue
        filtered.append(line)

    kept = [*str(stage["keywords"]).splitlines(), *filtered]
    options = str(stage.get("options", "")).rstrip()
    if options:
        kept.extend(["", options])
    kept.append(f"output movie cif {stage['prefix']}.cif")
    if bool(stage.get("needs_restart")):
        kept.append(f"dump {stage['prefix']}.grs")
    return "\n".join(kept).rstrip() + "\n"


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
            "message": "",
        }
        results.append(row)
        write_stage_results(calc_dir, results)

        try:
            for stale_path in (got_path, cif_path, restart_path):
                if stale_path.exists():
                    stale_path.unlink()
            if index:
                previous_restart = calc_dir / f"{stages[index - 1]['prefix']}.grs"
                if not previous_restart.exists():
                    raise FileNotFoundError(f"Previous stage restart is missing: {previous_restart.name}")
                gin_path.write_text(rewrite_restart(previous_restart.read_text(), stage, managed_heads))

            command = str(plan["gulp_command"]).replace("PREFIX", prefix)
            completed = subprocess.run(command, cwd=calc_dir, shell=True, check=False)
            data = parse_got(got_path)
            row.update({key: data.get(key) for key in STAGE_RESULT_FIELDS if key in data})
            if validate_atom_counts:
                validate_got_contract(data, expected_asu, expected_total)
            if completed.returncode != 0:
                raise RuntimeError(f"GULP command returned exit code {completed.returncode}")
            require_convergence = bool(stage.get("require_convergence", True))
            if require_convergence and not data.get("completed_normally"):
                raise RuntimeError("GULP did not report normal completion")
            converged = not bool(stage["is_optimisation"]) or data.get("gulp_status") == "optimisation_achieved"
            row["converged"] = converged
            if bool(stage["is_optimisation"]) and require_convergence and not converged:
                raise RuntimeError(str(data.get("gulp_status") or "optimisation did not converge"))
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
