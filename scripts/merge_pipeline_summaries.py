#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    try:
        return int(row.get("ID", "0")), row.get("name", "")
    except ValueError:
        return 0, row.get("name", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pyGulp POSCAR pipeline task summaries into one CSV.")
    parser.add_argument("output_dir", type=Path, help="Pipeline output directory containing summary_task_*.csv files.")
    parser.add_argument("--pattern", default="summary_task_*.csv", help="Input summary glob pattern.")
    parser.add_argument("-o", "--output", default="summary.csv", help="Merged output CSV path or name.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    summary_files = sorted(output_dir.glob(args.pattern))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {output_dir} with pattern {args.pattern}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for summary_file in summary_files:
        with summary_file.open(newline="") as fd:
            reader = csv.DictReader(fd)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    if fieldnames is None:
        raise ValueError("No CSV header found")

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = output_dir / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        sorted_rows = sorted(rows, key=sort_key)
        for row in sorted_rows:
            writer.writerow(row)

    import pandas as pd

    xlsx_output = output.with_suffix(".xlsx")
    pd.DataFrame(sorted_rows, columns=fieldnames).to_excel(xlsx_output, index=False, sheet_name="Summary")
    print(f"Merged {len(rows)} rows from {len(summary_files)} files into {output} and {xlsx_output}")


if __name__ == "__main__":
    main()
