from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list


Connection = tuple[int, int]


def _is_zero_image(image) -> bool:
    if image is None:
        return True
    return all(int(round(float(value))) == 0 for value in image)


def _validate_index_base(index_base: int) -> None:
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")


def _selected_indices(atoms: Atoms, molecule_tag: int | None) -> list[int]:
    if molecule_tag is None:
        return list(range(len(atoms)))

    tags = np.asarray(atoms.get_tags())
    return [int(i) for i in np.where(tags == molecule_tag)[0]]


def infer_natural_cutoff_connections(
    atoms: Atoms,
    molecule_tag: int | None = None,
    same_tag_only: bool = True,
    include_periodic_bonds: bool = True,
    index_base: int = 1,
    local_indexing: bool = False,
    mult: float = 1.1,
) -> list[Connection]:
    """Infer covalent-like ``connect`` pairs from ASE natural cutoffs."""

    _validate_index_base(index_base)

    selected = _selected_indices(atoms, molecule_tag)
    if not selected:
        raise ValueError(f"No atoms found for molecule_tag={molecule_tag}")

    selected_set = set(selected)
    tags = np.asarray(atoms.get_tags())

    if local_indexing:
        index_map = {old_idx: new_idx + index_base for new_idx, old_idx in enumerate(selected)}
    else:
        index_map = {old_idx: old_idx + index_base for old_idx in selected}

    cutoffs = natural_cutoffs(atoms, mult=mult)
    first, second, offsets = neighbor_list("ijS", atoms, cutoffs)

    pairs: set[Connection] = set()
    for i, j, offset in zip(first, second, offsets):
        i = int(i)
        j = int(j)
        if i == j:
            continue
        if i not in selected_set or j not in selected_set:
            continue
        if same_tag_only and tags[i] != tags[j]:
            continue
        if not include_periodic_bonds and not _is_zero_image(offset):
            continue

        a = index_map[i]
        b = index_map[j]
        pairs.add(tuple(sorted((int(a), int(b)))))

    return sorted(pairs)


def write_connections(connections: Iterable[Connection], output_path: str | Path) -> Path:
    """Write pairs to a GULP-compatible ``connections`` file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fd:
        for atom1, atom2 in connections:
            fd.write(f"connect {int(atom1)} {int(atom2)}\n")

    return output_path


def write_natural_cutoff_connections(
    atoms: Atoms,
    output_path: str | Path,
    molecule_tag: int | None = None,
    same_tag_only: bool = True,
    include_periodic_bonds: bool = True,
    local_indexing: bool | None = None,
    mult: float = 1.1,
) -> list[Connection]:
    """Infer natural-cutoff bonds and write a GULP ``connections`` file."""

    if local_indexing is None:
        local_indexing = molecule_tag is not None

    connections = infer_natural_cutoff_connections(
        atoms=atoms,
        molecule_tag=molecule_tag,
        same_tag_only=same_tag_only,
        include_periodic_bonds=include_periodic_bonds,
        index_base=1,
        local_indexing=local_indexing,
        mult=mult,
    )
    write_connections(connections, output_path)
    return connections


def set_contiguous_molecule_tags(atoms: Atoms, atoms_per_molecule: int) -> None:
    """Assign ASE tags for structures ordered as molecule blocks."""

    if atoms_per_molecule <= 0:
        raise ValueError("atoms_per_molecule must be positive")
    if len(atoms) % atoms_per_molecule != 0:
        raise ValueError("Number of atoms is not divisible by atoms_per_molecule")

    n_molecules = len(atoms) // atoms_per_molecule
    atoms.set_tags(np.repeat(np.arange(n_molecules), atoms_per_molecule))


def molecule_tags_from_connections(atoms: Atoms, connections: Iterable[Connection]) -> np.ndarray:
    """Build molecule tags as graph connected components from 0-based pairs."""

    adjacency = [set() for _ in range(len(atoms))]
    for atom1, atom2 in connections:
        adjacency[int(atom1)].add(int(atom2))
        adjacency[int(atom2)].add(int(atom1))

    tags = np.full(len(atoms), -1, dtype=int)
    tag = 0

    for start in range(len(atoms)):
        if tags[start] != -1:
            continue

        queue: deque[int] = deque([start])
        tags[start] = tag

        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if tags[neighbour] == -1:
                    tags[neighbour] = tag
                    queue.append(neighbour)

        tag += 1

    return tags


def infer_molecule_tags_natural_cutoffs(
    atoms: Atoms,
    include_periodic_bonds: bool = True,
    mult: float = 1.1,
) -> np.ndarray:
    """Infer molecule-like connected components from natural-cutoff bonds."""

    connections = infer_natural_cutoff_connections(
        atoms=atoms,
        molecule_tag=None,
        same_tag_only=False,
        include_periodic_bonds=include_periodic_bonds,
        index_base=0,
        local_indexing=False,
        mult=mult,
    )
    return molecule_tags_from_connections(atoms, connections)
