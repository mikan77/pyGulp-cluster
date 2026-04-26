#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from ase.io import read
from pygulp.molecule.connections import (
    infer_molecule_tags_natural_cutoffs,
    infer_natural_cutoff_connections,
    set_contiguous_molecule_tags,
    write_connections,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a GULP connections file from a POSCAR/CIF using ASE natural cutoffs."
    )
    parser.add_argument("structure", help="Input structure readable by ASE, e.g. POSCAR or CIF.")
    parser.add_argument("-f", "--format", default=None, help="ASE input format, e.g. vasp or cif.")
    parser.add_argument("-o", "--output", default="connections", help="Output connections file.")
    parser.add_argument(
        "--atoms-per-molecule",
        type=int,
        default=None,
        help="Assign tags as contiguous molecule blocks of this size.",
    )
    parser.add_argument(
        "--infer-molecule-tags",
        action="store_true",
        help="Assign tags as natural-cutoff connected components before writing connections.",
    )
    parser.add_argument(
        "--molecule-tag",
        type=int,
        default=None,
        help="Write only the molecule with this tag. With --atoms-per-molecule, defaults to 0.",
    )
    parser.add_argument(
        "--global-indexing",
        action="store_true",
        help="Keep original atom numbers instead of renumbering the selected molecule to 1..N.",
    )
    parser.add_argument(
        "--include-intermolecular",
        action="store_true",
        help="Keep pairs across different ASE tags.",
    )
    parser.add_argument(
        "--exclude-periodic-bonds",
        action="store_true",
        help="Drop neighbours found through non-zero periodic images. Periodic bonds are kept by default.",
    )
    parser.add_argument(
        "--natural-mult",
        type=float,
        default=1.1,
        help="ASE natural cutoff multiplier.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print molecule tag sizes and compositions after tags are assigned.",
    )
    args = parser.parse_args()

    atoms = read(args.structure, format=args.format)
    atoms.set_pbc([True, True, True])

    include_periodic_bonds = True
    if args.exclude_periodic_bonds:
        include_periodic_bonds = False

    molecule_tag = args.molecule_tag
    if args.atoms_per_molecule is not None and args.infer_molecule_tags:
        parser.error("Use either --atoms-per-molecule or --infer-molecule-tags, not both.")

    if args.atoms_per_molecule is not None:
        set_contiguous_molecule_tags(atoms, args.atoms_per_molecule)
        if molecule_tag is None:
            molecule_tag = 0

    if args.infer_molecule_tags:
        tags = infer_molecule_tags_natural_cutoffs(
            atoms,
            include_periodic_bonds=include_periodic_bonds,
            mult=args.natural_mult,
        )
        atoms.set_tags(tags)
        if molecule_tag is None:
            molecule_tag = 0

    if args.summary and (args.atoms_per_molecule is not None or args.infer_molecule_tags):
        tags = np.asarray(atoms.get_tags())
        for tag in sorted(set(tags.tolist())):
            indices = np.where(tags == tag)[0]
            formula = dict(Counter(atoms[int(index)].symbol for index in indices))
            print(
                f"tag {tag}: size={len(indices)} formula={formula} "
                f"indices_1based={(indices + 1).tolist()}"
            )

    local_indexing = molecule_tag is not None and not args.global_indexing

    connections = infer_natural_cutoff_connections(
        atoms=atoms,
        molecule_tag=molecule_tag,
        same_tag_only=not args.include_intermolecular,
        include_periodic_bonds=include_periodic_bonds,
        index_base=1,
        local_indexing=local_indexing,
        mult=args.natural_mult,
    )
    write_connections(connections, args.output)

    if molecule_tag is None:
        scope = "full structure"
    elif local_indexing:
        scope = f"molecule tag {molecule_tag}, local indexing"
    else:
        scope = f"molecule tag {molecule_tag}, global indexing"

    print(f"Wrote {len(connections)} connections for {scope} to {args.output}")


if __name__ == "__main__":
    main()
