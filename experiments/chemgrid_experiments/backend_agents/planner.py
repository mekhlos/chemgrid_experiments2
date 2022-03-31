from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from chemgrid_game.chemistry.mol_chemistry import Action
from chemgrid_game.mol_archive import MolArchive

from experiments.chemgrid_experiments.graph_search.search import GraphSearchBase
from experiments.chemgrid_experiments.graph_search.search import GraphSearchExhaustive
from experiments.chemgrid_experiments.graph_search.search import GraphSearchHeuristic
from experiments.chemgrid_experiments.mol_graph import MolGraph


class Planner:
    def __init__(
            self,
            archive: MolArchive,
            search_class: Type[GraphSearchBase],
            explore_search_class: Type[GraphSearchBase] = GraphSearchExhaustive,
            return_best_if_fail: bool = False,
            **search_params
    ):
        self.archive = archive
        self.search_class = search_class
        self.explore_search_class = explore_search_class
        self.search_params = search_params
        self.return_best_if_fail = return_best_if_fail
        self.counter = {}

    def record_mols(self, path: List[Action], archive: MolGraph):
        for action in path:
            for parent_hash in action.operands:
                if parent_hash not in self.counter:
                    self.counter[parent_hash] = 0
                self.counter[parent_hash] += 1
                self.archive.add(archive[action.res])

    def plan(self, inventory: List[int], target_mol_id: Optional[int], explore=False) -> Dict:
        res = {}
        inventory = [self.archive[i] for i in inventory]
        archive = MolGraph(initial_mols=inventory)
        res["archive"] = archive

        if explore:
            target_mol = None if target_mol_id is None else self.archive[target_mol_id]
            gs = self.explore_search_class(archive, **self.search_params, target_mol=target_mol)
        else:
            target_mol = self.archive[target_mol_id]
            gs = self.search_class(archive, **self.search_params, target_mol=target_mol)

        success = gs.search(n_steps=self.search_params["n_steps"], target_mol=target_mol)
        if success:
            res["success"] = True
            res["path"] = archive.get_path(hash(target_mol))
            self.record_mols(res["path"], archive)
        else:
            res["success"] = False
            if self.return_best_if_fail and isinstance(gs, GraphSearchHeuristic):
                mol = archive[gs.inventory[0]]
                res["path"] = archive.get_path(hash(mol))
                self.record_mols(res["path"], archive)

        return res
