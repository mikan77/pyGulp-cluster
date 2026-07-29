# pyGulp Cluster

`pyGulp-cluster` is a cluster-oriented version of `pyGulp` for batch preparation
and multi-launch of GULP calculations through `SLURM`.

This version is designed for:
- preparing one `CalcFold` per structure
- submitting all calculations through `sbatch`
- limiting the number of simultaneously active jobs with `--max-parallel`
- collecting compact summary tables after or during the workflow

It is not intended as a local direct-run wrapper around `gulp`. The main mode is
cluster submission.

## What This Version Does

For each input structure the program:
- automatically reads CIF and POSCAR/VASP structures
- determines symmetry with spglib, builds a conventional cell, validates its
  asymmetric unit, and writes the detected space group to GULP
- falls back to the full P1 structure when symmetry preparation is not reliable
- infers molecular components and generates diagnostic GULP `connect` records
- creates a working directory with `CalcFold`
- writes `ginput1.gin`
- writes `job.sh`
- copies the selected `.lib` file into `CalcFold`
- submits jobs through `sbatch`
- keeps at most `N` simultaneously submitted jobs when `--max-parallel N` is used
- writes `summary.csv`, `summary.jsonl`, and `summary.xlsx`
- exports validated, Mercury-readable final CIF files as `relaxed_cifs/<ID>.cif`
- counts molecules in the final expanded structure

## Supported Force Fields

When `--library` is supplied, the active force field is determined from the library filename:
- `reaxff*.lib` -> `reaxff`
- `gfnff*.lib` -> `gfnff`

`dreiding*.lib` is intentionally rejected in this build.

For `reaxff` and `gfnff`, atomic coordinate lines in `ginput1.gin` are written
without `core`. Example:

```text
C 0.5082000 0.7791000 0.6671000 0.00000
```

## Required Inputs

You need:

1. A directory with structures.
   Supported by default:
   - `POSCAR*`
   - `*POSCAR*`
   - `*.vasp`
   - `*.poscar`
   - `*.cif`

2. `keyword.in`
   This file contains the GULP keywords section placed before coordinates.

3. `options.in`
   This file contains the GULP options section placed after coordinates.

4. Optional: a force-field library file passed through `--library`.

5. `SLURM` tools available in the environment:
   - `sbatch`
   - `squeue`
   - `sacct`

6. `gulp` available on the compute nodes through your job script environment.

## Where The `.lib` File Must Be

If `--library` is used in the binary workflow, the `.lib` file must be placed next to the binary.

Example:

```text
dist/
├── pygulp-cluster
├── reaxff_general.lib
└── gfnff.lib
```

Then launch with:

```bash
./dist/pygulp-cluster /path/to/structures --library reaxff_general.lib
```

If `--library` is omitted, no `.lib` file is copied and no `library ...` line is
written to `ginput1.gin`.

When running from source with `scripts/run_poscar_folder.py`, the program also
accepts the library in the project root as a convenience for development.

## What Goes Into `keyword.in`

`keyword.in` is inserted at the top of `ginput1.gin`.

Example:

```text
opti gradient conp conse qok c6 conp prop gfnff gwolf noauto
gfnff_scale 0.8 1.343 0.727 1.0 2.859
maths mrrr
pressure 0 GPa
```

Or for a ReaxFF-style run:

```text
opti conp reaxff
```

Use this file for:
- GULP keywords
- force-field-specific keyword lines
- pressure and math options that belong before coordinates

## What Goes Into `options.in`

`options.in` is inserted after coordinates.

Example:

```text
maxcycle 500
gtol 0.000001
```

Use this file for:
- convergence settings
- cycle limits
- post-coordinate GULP options

The program automatically appends:
- `library <selected_library>`
- generated `connect ...` lines for full P1 inputs unless `--no-connections` is used
- `output movie cif relaxed.cif`

Full-cell `connect` records are intentionally omitted when a reduced asymmetric
unit is written because their atom indices do not address ASU sites.

## Job Script Handling

If a `job.sh` file exists at the path passed to `--job-template` or in the
current working directory with the default name `job.sh`, it is used as a
template.

If no template file is found, the program writes a built-in `SLURM` script.

The generated script always gets a per-structure job name and runs:

```bash
gulp < ginput1.gin > ginput1.got
```

You can also customize:
- `--job-time`
- `--job-cpus`
- `--job-partition`
- `--job-account`
- `--job-module`
- `--job-extra-line`

## Main Launch Logic

The workflow is:

1. Prepare all structure folders.
2. Generate `ginput1.gin` and `job.sh`.
3. Submit the first `N` jobs through `sbatch`.
4. Track submitted jobs by `job_id`.
5. As soon as one job finishes, submit the next pending job.
6. Continue until all jobs are done.

`--max-parallel` controls the width of this submission window.

Example:
- `--max-parallel 32`
  means at most 32 submitted jobs are kept active at once
- when one job finishes, another `job.sh` is submitted in its place

## Command-Line Interface

Main arguments:
- `input_dir`
- `--output-dir`
- `--keywords-file`
- `--options-file`
- `--library`
- `--max-parallel`
- `--prepare-only`
- `--collect-only`

Defaults:
- `--output-dir pygulp_runs`
- `--keywords-file keyword.in`
- `--options-file options.in`
- `--max-parallel 32`
- `--format auto`
- `--symprec 0.05`
- symmetry preparation enabled

## Launch Examples

### 1. Prepare only

```bash
python3 scripts/run_poscar_folder.py /path/to/structures \
  --library reaxff_general.lib \
  --prepare-only
```

This creates all working folders and input files, but does not submit jobs.

### 2. Submit with a 32-job window

```bash
python3 scripts/run_poscar_folder.py /path/to/structures \
  --library reaxff_general.lib \
  --max-parallel 32
```

### 3. CIF input

```bash
python3 scripts/run_poscar_folder.py /path/to/cifs \
  --pattern '*.cif' \
  --library gfnff.lib \
  --max-parallel 16
```

### 4. Rebuild summary from existing results

```bash
python3 scripts/run_poscar_folder.py /path/to/structures \
  --output-dir pygulp_runs \
  --collect-only
```

### 5. Run through the binary

```bash
./dist/pygulp-cluster /path/to/structures \
  --library reaxff_general.lib \
  --max-parallel 32
```

## Output Layout

For each structure:

```text
<output_dir>/
└── 00001_structure_name/
    ├── input.cif
    ├── standardized_full.cif
    ├── asymmetric_unit.cif
    ├── symmetry.json
    ├── keyword.in
    ├── options.in
    ├── generated_options.in
    ├── connections
    ├── connections_tag_0
    ├── molecule_summary.json
    └── CalcFold/
        ├── ginput1.gin
        ├── ginput1.got
        ├── relaxed.cif
        ├── job.sh
        ├── <selected_library>.lib  # only when --library is supplied
        ├── slurm_<jobid>.out
        └── slurm_<jobid>.err
```

In the output root:
- `summary.csv`
- `summary.jsonl`
- `summary.xlsx`
- `structure_manifest.json`
- `relaxed_cifs/<ID>.cif`
- `dispatcher.log`

## Summary Columns

The compact summary contains:
- `ID`
- `name`
- `status`
- `formula`
- `n_atoms_input`
- `n_atoms_conventional`
- `n_atoms_asu`
- `n_molecules_final`
- `input_spacegroup`
- `input_spacegroup_number`
- `gulp_spacegroup`
- `gulp_spacegroup_number`
- `final_spacegroup`
- `final_spacegroup_number`
- `symmetry_operations`
- `symmetry_fallback`
- `energy_initial_ev`
- `energy_final_ev`
- `volume`
- `runtime_seconds`
- `density_g_cm3`
- `energy_initial_ev_per_atom`
- `energy_final_ev_per_atom`
- `cif_file`
- `cif_status`

Input paths and technical paths such as `gin`, `got`, `job_script`, and
`submitted_job_id` are intentionally not stored in the summary.

## Building The Binary

The project contains a build script:

```bash
sh scripts/build_binary.sh
```

This uses `PyInstaller` and writes the binary to:

```text
dist/pygulp-cluster
```

If `PyInstaller` is not installed:

```bash
python3 -m pip install --user pyinstaller
```

Then build:

```bash
cd /Users/mihailaveranov/Desktop/pip/my_code/pyGulp-cluster
sh scripts/build_binary.sh
```

Check the result:

```bash
./dist/pygulp-cluster --help
```

After build, place the required `.lib` next to the binary before running.

## Development Entry Points

You can run either:

```bash
python3 scripts/run_poscar_folder.py --help
```

or, after installation as a package:

```bash
pygulp-cluster --help
```

Run the symmetry/XLSX self-check with:

```bash
python3 scripts/check_cluster_symmetry.py
```

## Notes

- This build is cluster-oriented and assumes `SLURM`.
- Symmetry is detected from the input and used for GULP by default. Use
  `--no-symmetry` to keep the full input structure in P1.
- `--collect-only` also rebuilds XLSX and the numbered final-CIF directory.
- `gulp` itself is not bundled into the binary.
- `.lib` files are not bundled into the binary.
- If `ginput1.got` exists but the optimization did not converge, the final
  `status` is derived from the parsed GULP output, not only from `SLURM`.
