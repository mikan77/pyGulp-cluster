#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pygulp.cluster import SUMMARY_FIELDS, prepare_symmetry, render_gulp_input, write_summary


def main() -> None:
    conventional, asu, metadata = prepare_symmetry(
        bulk("Si", "diamond", a=5.43, cubic=True),
        symprec=0.05,
    )
    assert len(conventional) == 8
    assert len(asu) == 1
    assert metadata["gulp_spacegroup_number"] == 227
    assert metadata["symmetry_fallback"] is False

    gin = render_gulp_input(
        asu,
        keywords="single",
        options="output movie cif relaxed.cif",
        spacegroup_number=227,
    )
    assert "\nspacegroup\n227\n" in gin

    with TemporaryDirectory() as directory:
        row = {field: None for field in SUMMARY_FIELDS}
        row.update({"ID": 1, "name": "silicon", "status": "prepared"})
        write_summary(Path(directory), [row])
        assert all((Path(directory) / f"summary.{suffix}").exists() for suffix in ("csv", "jsonl", "xlsx"))

    print("symmetry pipeline self-check passed")


if __name__ == "__main__":
    main()
