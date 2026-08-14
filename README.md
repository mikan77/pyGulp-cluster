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
- optionally runs several GULP stages sequentially under one SLURM job ID
- copies the selected `.lib` file into `CalcFold`
- submits jobs through `sbatch`
- keeps at most `N` simultaneously submitted jobs when `--max-parallel N` is used
- writes `summary.csv`, `summary.jsonl`, and `summary.xlsx`
- exports validated, Mercury-readable final CIF files as `relaxed_cifs/<ID>.cif`
- counts molecules in the final expanded structure

The symmetry-constrained rigid mode is selected with `mode: rigid_gfnff_symmetry`.
The old `rigid_gfnff` mode has been removed. The new mode expands the ASU to a
full P1 cell for pGFNFF gradient calculations, projects gradients back to
independent rigid molecules, and preserves the selected space group during
every accepted step.

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

2. `keyword.in` for the default single-stage mode.
   This file contains the GULP keywords section placed before coordinates.

3. `options.in` for the default single-stage mode.
   This file contains the GULP options section placed after coordinates.

4. Optional: a force-field library file passed through `--library`.

5. `SLURM` tools available in the environment:
   - `sbatch`
   - `squeue`
   - `sacct`

6. `gulp` available on the compute nodes through your job script environment.

For multi-stage mode, `--stages-file` replaces `keyword.in`. `options.in` is
only used as the default for stages that omit their own `options` field.

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
./dist/pygulp-cluster/pygulp-cluster /path/to/structures --library reaxff_general.lib
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

## Multi-Stage Mode

Pass a YAML file to run several calculations sequentially under one SLURM job ID:

```bash
pygulp-cluster /path/to/structures \
  --stages-file configs/stages_relax_static.yaml \
  --library reaxff_general.lib
```

The included profile performs:

1. atomic relaxation with a fixed cell (`conv`)
2. atomic and cell relaxation (`conp`)
3. an optional final single-point calculation

Each non-final stage must be an optimisation. GULP passes its updated geometry
and cell to the next stage through a native `.grs` restart file. For these
stages pyGulp automatically writes the restart after every optimisation cycle
using `dump every 1`, so a normally completed run with
`require_convergence: false` can continue from the last saved geometry after
`max_function_calls` or `Too many failed attempts to optimise`. The latter is
accepted only when the restart file exists; syntax errors and missing restart
files remain fatal. By default the next stage requires `Optimisation achieved`;
set `require_convergence: false` to allow a non-converged but restartable stage.

By default each stage does not enforce atom-count checks from `.got` output. Use
`validate_atom_counts: true` in a stage block only if you want strict checking.

If your workflow needs full validation for all stages, either:
- set top-level `validate_atom_counts: true` in the YAML file, or
- set it per stage explicitly.

Example configuration:

```yaml
final_symmetry:
  required: true
  symprec: 0.05
  angle_tolerance: 5.0

validate_atom_counts: true   # optional, applies to all stages by default
stages:
  - name: fixed_cell
    symmetry_mode: "off"
    require_convergence: false
    validate_atom_counts: true
    keywords: |
      opti conj reaxff conv qiter spat
    options: |
      gtol 1e-3
      maxcyc 500
      stepmx 0.05

  - name: variable_cell
    symmetry_mode: auto
    require_convergence: false
    validate_atom_counts: true
    keywords: |
      opti conj reaxff conp qiter spat
    options: |
      gtol 5e-4
      maxcyc 1000
      stepmx 0.02
```

For ordinary GULP stages, `symmetry_mode: "off"` relaxes the full structure in
P1. `symmetry_mode: auto` reads the previous stage CIF, determines its current
symmetry, builds a new asymmetric unit, and writes the next GULP input from
that structure. The default is `auto`. The rigid GFNFF mode always preserves
symmetry and cannot use `off`.

After the last stage the final `relaxed.cif` is always checked for symmetry.
`P1` is a valid result; a missing or unreadable final CIF is an error when
`final_symmetry.required: true`. The check does not modify the final geometry.
The YAML values control the tolerances used for this final analysis.

Without `--stages-file`, the single-stage input workflow is unchanged; the
final CIF symmetry check uses the CLI tolerances.

## Job Script Handling

If a `job.sh` file exists at the path passed to `--job-template` or in the
current working directory with the default name `job.sh`, it is used as a
template.

If no template file is found, the program writes a built-in `SLURM` script.

The generated script always gets a per-structure job name and runs:

```bash
gulp < ginput1.gin > ginput1.got
```

In multi-stage mode this command is replaced with the internal stage runner.
SBATCH settings, module commands, `ulimit`, and other template lines are retained.

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
- `--stages-file`
- `--force`
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
./dist/pygulp-cluster/pygulp-cluster /path/to/structures \
  --library reaxff_general.lib \
  --max-parallel 32
```

### 6. Fixed-cell, variable-cell, and static stages

```bash
./dist/pygulp-cluster/pygulp-cluster /path/to/structures \
  --stages-file configs/stages_relax_static.yaml \
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
    ├── asymmetric_unit.xyz
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
        ├── stage_plan.json              # multi-stage mode
        ├── stage_results.json           # multi-stage mode
        ├── stages.csv                    # multi-stage mode
        ├── 01_fixed_cell.gin/.got/.grs  # multi-stage mode
        ├── 02_variable_cell.gin/.got/.grs
        ├── 03_static.gin/.got
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
- `stages.csv` when multi-stage results exist

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
- `completed_stages`
- `total_stages`
- `failed_stage`

Input paths and technical paths such as `gin`, `got`, `job_script`, and
`submitted_job_id` are intentionally not stored in the summary.

## Building The Binary

The project contains a build script:

```bash
sh scripts/build_binary.sh
```

This uses `PyInstaller` and writes the binary to:

```text
dist/pygulp-cluster/pygulp-cluster
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
./dist/pygulp-cluster/pygulp-cluster --help
```

If you are using multi-stage mode, always use the onedir build. Onefile launchers use temporary
`_MEI...` paths and are not reliable for SLURM child-process execution.

For the COLLECT error:

```bash
rm -rf build dist
sh scripts/build_binary.sh
```

If you still see `Resource '.../dist/pygulp-cluster' is not a valid file`, wipe build artifacts and rebuild again:

```bash
rm -rf build dist
sh scripts/build_binary.sh
```

For multi-stage runs (`--stages-file`), an onedir build is required: onefile launchers place temp executables under `/tmp/_MEI...` and SLURM jobs can lose that path.

After build, verify:

```bash
ls -l dist/pygulp-cluster
# should contain:
# - pygulp-cluster (executable)
# - _internal/*
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

Run the restart and multi-stage self-check with:

```bash
PYTHONPATH=src python3 scripts/check_multistage_pipeline.py
```

## Notes

- This build is cluster-oriented and assumes `SLURM`.
- Symmetry is detected from the input and used for GULP by default. Use
  `--no-symmetry` to keep the full input structure in P1.
- `--collect-only` also rebuilds XLSX and the numbered final-CIF directory.
- Existing `.got`, `relaxed.cif`, and stage results are preserved unless
  `--force` is explicitly supplied.
- When recursive input search is enabled, a nested output directory is excluded
  so generated CIF files cannot be submitted as new structures.
- `gulp` itself is not bundled into the binary.
- `.lib` files are not bundled into the binary.
- If `ginput1.got` exists but the optimization did not converge, the final
  `status` is derived from the parsed GULP output, not only from `SLURM`.
