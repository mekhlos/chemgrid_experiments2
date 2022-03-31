from collections import deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from PIL import Image
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_backend import Action
from chemgrid_game.mol_archive import MolArchive
from matplotlib import pyplot as plt

from experiments.chemgrid_experiments.graph_search import graph_generation

Edge = Tuple[str, Molecule, Molecule]


class MolGraph(MolArchive):
    def __init__(self, initial_mols: List[Molecule] = ()):
        self.parent_actions: Dict[int, Set[Action]] = {}
        self.best_parent_action: Dict[int, Action] = {}
        self.children: Dict[int, Set[int]] = {}
        self.best_children: Dict[int, Set[int]] = {}
        self.stats = {"new": 0, "old": 0, "shorter_path": 0}
        self._min_steps: Dict[int, int] = {}
        self._min_depths: Dict[int, int] = {}
        super().__init__(initial_mols)

    def add(
            self,
            mol: Molecule,
            action: Optional[Action] = Action()
    ) -> bool:
        mol_hash = hash(mol)
        if mol_hash not in self:
            self[mol_hash] = mol
            self.best_parent_action[mol_hash] = action
            self.parent_actions[mol_hash] = {action}
            self.stats["new"] += 1
            self.children[mol_hash] = set()
            self.best_children[mol_hash] = set()
            for parent_id in action.operands:
                self.children[parent_id].add(mol_hash)
                self.best_children[parent_id].add(mol_hash)

            return True

        else:
            self.parent_actions[mol_hash].add(action)
            new_steps = self.get_min_steps(parent_hashes=action.operands)
            old_steps = self.get_min_steps(mol_hash)
            if new_steps < old_steps:
                self.stats["shorter_path"] += 1
                old_parent_action = self.best_parent_action[mol_hash]
                for old_parent_hash in old_parent_action.operands:
                    self.best_children[old_parent_hash].remove(mol_hash)
                for new_parent_hash in action.operands:
                    self.best_children[new_parent_hash].add(mol_hash)

                self.best_parent_action[mol_hash] = action

            self.stats["old"] += 1

        return False

    def get_min_depths(self, mol_hash: int = None, parent_hashes=None, use_cached=False):
        if mol_hash is not None:
            if use_cached and mol_hash in self._min_depths:
                return self._min_depths[mol_hash]

            parent_hashes = self.best_parent_action[mol_hash].operands

        self._min_depths[mol_hash] = 1 + max(self.get_min_depths(pid) for pid in parent_hashes)

        return self._min_depths[mol_hash]

    def get_min_steps(self, mol_hash: int = None, parent_hashes=None, use_cached=False):
        if mol_hash is not None:
            if use_cached and mol_hash in self._min_steps:
                return self._min_steps[mol_hash]

            parent_hashes = self.best_parent_action[mol_hash].operands

        self._min_steps[mol_hash] = 1 + self.count_ancestors(parent_hashes=parent_hashes, include_roots=False)

        return self._min_steps[mol_hash]

    def count_ancestors(self, mol_hash: int = None, parent_hashes=None, include_roots: bool = True) -> int:
        if parent_hashes is None:
            parent_hashes, _ = self.best_parent_action[mol_hash]

        q = deque()
        q.extend(parent_hashes)
        ancestors = set()
        while len(q) > 0:
            mol_hash = q.popleft()
            parents = self.best_parent_action[mol_hash].operands
            if include_roots or len(parents) > 0:
                ancestors.add(mol_hash)
            q.extend(parents)

        return len(ancestors)

    def convert_action_ids_to_mols(self, action: Action) -> Action:
        operands = tuple([self[parent_hash] for parent_hash in action.operands])
        res = self[action.res] if action.res is not None else None
        return Action(action.op, operands, action.params, res)

    def get_path(self, mol_hash: int, actual_mols: bool = False) -> List[Action]:
        path = []
        queue = deque()
        queue.append(mol_hash)
        while len(queue) > 0:
            mol_hash = queue.popleft()
            action = self.best_parent_action[mol_hash]
            if action.op != "noop":
                if action in path:
                    path.remove(action)
                path.append(action)
            queue.extend(action.operands)

        path = path[::-1]

        if actual_mols:
            path = [self.convert_action_ids_to_mols(action) for action in path]

        return path

    def get_edges(self, mol_hash: int, actual_mols: bool = False) -> List[Edge]:
        path = self.get_path(mol_hash, actual_mols)
        edges = []
        for i, action in enumerate(path):
            for parent in action.operands:
                edges.append((action.op, parent, action.res))

        return edges

    def get_score(self, mol_hash: int, use_depth: bool = False, cached=False) -> float:
        if use_depth:
            x = self.get_min_depths(mol_hash, use_cached=cached)
        else:
            x = self.get_min_steps(mol_hash, use_cached=cached)

        return np.sum(self[mol_hash].atoms > 0) / np.sqrt(1 + x)

    def plot_graph(self, mol_hash: int = None, size=None):
        edges = self.get_edges(mol_hash, actual_mols=True)
        graph = graph_generation.edge_list_to_graph(edges)
        p = graph_generation.save_graph_to_img(graph)
        if size is None:
            m = n = int(np.sqrt(len(edges))) * 2
            size = (m, n)
        fig, ax = plt.subplots(figsize=size)

        with Image.open(p) as im:
            ax.imshow(im)
            ax.set_axis_off()

        plt.show()
        return fig, ax

    def reset(self):
        super().reset()
        self.parent_actions.clear()
        self.best_parent_action.clear()
        self.children.clear()
        self.best_children.clear()
        self._min_steps.clear()
        self._min_depths.clear()
        self.stats.clear()
