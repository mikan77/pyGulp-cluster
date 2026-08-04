#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pygulp.stages import execute_stage_plan, parse_got, rewrite_restart, validate_got_contract


def main() -> None:
    restart = """opti conj reaxff conv qiter spat
parallel better_scaling
title
ASE calculation
end

cell
10 10 10 90 90 90
frac
C 0 0 0 0

spacegroup
1

maxcyc 100
output movie cif old.cif
dump old.grs
"""
    stage = {
        "prefix": "02_variable_cell",
        "keywords": "opti conj reaxff conp qiter spat\nparallel better_scaling",
        "options": "maxcyc 500\nstepmx 0.02",
        "needs_restart": True,
    }
    rewritten = rewrite_restart(restart, stage, {"maxcyc", "stepmx", "output", "dump"})
    assert "conp" in rewritten and "conv" not in rewritten
    assert "maxcyc 100" not in rewritten and "maxcyc 500" in rewritten
    assert "output movie cif 02_variable_cell.cif" in rewritten
    assert "dump 02_variable_cell.grs" in rewritten

    with TemporaryDirectory() as directory:
        calc_dir = Path(directory)
        got = calc_dir / "stage.got"
        got.write_text(
            "Number of irreducible atoms/shells = 50\n"
            "Total number atoms/shells = 100\n"
            "Total lattice energy = -10.5 eV\n"
            "Final Gnorm = 0.0001\n"
            "Optimisation achieved\n"
            "GULP has completed\n"
        )
        data = parse_got(got)
        validate_got_contract(data, expected_asu=50, expected_total=100)
        assert data["gulp_status"] == "optimisation_achieved"
        assert data["energy_final_ev"] == -10.5

        fake_gulp = calc_dir / "fake_gulp.py"
        fake_gulp.write_text(
            "import shutil, sys\n"
            "from pathlib import Path\n"
            "prefix = sys.argv[1]\n"
            "Path(prefix + '.got').write_text(\n"
            "    'Number of irreducible atoms/shells = 1\\n'\n"
            "    'Total number atoms/shells = 1\\n'\n"
            "    'Total lattice energy = -1.0 eV\\n'\n"
            "    + ('' if 'static' in prefix else 'Maximum number of function calls has been reached\\n')\n"
            "    + 'GULP has completed\\n'\n"
            ")\n"
            "Path(prefix + '.cif').write_text('data_test\\n')\n"
            "if 'static' not in prefix:\n"
            "    shutil.copy2(prefix + '.gin', prefix + '.grs')\n"
        )
        prefixes = ("01_fixed", "02_variable", "03_static")
        first_gin = calc_dir / f"{prefixes[0]}.gin"
        first_gin.write_text(restart.replace("old.cif", f"{prefixes[0]}.cif").replace("old.grs", f"{prefixes[0]}.grs"))
        stages = [
            {
                "name": prefixes[0],
                "prefix": prefixes[0],
                "keywords": "opti conj reaxff conv",
                "options": "maxcyc 100",
                "is_optimisation": True,
                "require_convergence": False,
                "needs_restart": True,
            },
            {
                "name": prefixes[1],
                "prefix": prefixes[1],
                "keywords": "opti conj reaxff conp",
                "options": "maxcyc 100",
                "is_optimisation": True,
                "require_convergence": False,
                "needs_restart": True,
            },
            {
                "name": prefixes[2],
                "prefix": prefixes[2],
                "keywords": "single reaxff",
                "options": "",
                "is_optimisation": False,
                "needs_restart": False,
            },
        ]
        plan = {
            "n_atoms_asu": 1,
            "n_atoms_conventional": 1,
            "gulp_command": f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_gulp))} PREFIX",
            "stages": stages,
        }
        plan_path = calc_dir / "stage_plan.json"
        plan_path.write_text(json.dumps(plan))
        assert execute_stage_plan(plan_path) == 0
        results = json.loads((calc_dir / "stage_results.json").read_text())
        assert [item["status"] for item in results] == [
            "completed_nonconverged",
            "completed_nonconverged",
            "success",
        ]
        assert "conp" in (calc_dir / "02_variable.gin").read_text()
        assert "single" in (calc_dir / "03_static.gin").read_text()
        assert (calc_dir / "ginput1.got").exists() and (calc_dir / "relaxed.cif").exists()

    print("multistage pipeline self-check passed")


if __name__ == "__main__":
    main()
