#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATTERNS = ("POSCAR*", "*POSCAR*", "*.vasp", "*.poscar")


def collect_poscars(input_dir: Path, recursive: bool, patterns: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()

    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in iterator:
            if path.is_file():
                paths.add(path.resolve())

    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a text file with POSCAR paths from a directory.")
    parser.add_argument("input_dir", help="Directory containing POSCAR/VASP files.")
    parser.add_argument("-o", "--output", default="poscars.txt", help="Output list file.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories recursively.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern to include. Can be passed multiple times.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    poscars = collect_poscars(input_dir, args.recursive, patterns)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fd:
        for path in poscars:
            fd.write(f"{path}\n")

    print(f"Wrote {len(poscars)} POSCAR paths to {output}")


if __name__ == "__main__":
    main()
