from typing import List

import numpy as np

from experiments.chemgrid_experiments.graph_search.search import GraphSearchRandom
from experiments.chemgrid_experiments.mol_graph import MolGraph
from experiments.chemgrid_experiments.new_chemistry.chemistry_wrapper import ChemistryWrapper
from experiments.chemgrid_experiments.new_chemistry.mols import Molecule


def generate_random_mols(
        initial_mols: List[Molecule],
        n: int,
        n_steps: int,
        rng=None,
        min_size: int = 0,
        max_size: int = np.inf
) -> List[Molecule]:
    archive = MolGraph(initial_mols)
    chemistry = ChemistryWrapper()
    search = GraphSearchRandom(archive, chemistry=chemistry, rng=rng)
    search.search(n_steps)
    res = []
    for h in archive.hashes[::-1]:
        mol = archive[h]
        if min_size <= np.sum(mol.atoms) <= max_size:
            res.append(mol)
            if len(res) == n:
                break

    return res
