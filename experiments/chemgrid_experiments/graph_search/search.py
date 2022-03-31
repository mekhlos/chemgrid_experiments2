import abc
from typing import Optional

import numpy as np
from chemgrid_game.chemistry.mol_chemistry import ChemistryWrapper
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.chemistry.utils import generate_random_mol
from chemgrid_game.game_backend import Action
from chemgrid_game.utils import setup_logger
from tqdm import tqdm

from experiments.chemgrid_experiments.graph_search import heuristic_tools
from experiments.chemgrid_experiments.graph_search.priority_queue import PriorityQueue
from experiments.chemgrid_experiments.mol_graph import MolGraph


class GraphSearchBase:
    def __init__(
            self,
            archive: MolGraph,
            logger_level: str = "INFO",
            use_tqdm: bool = True,
            log_freq: int = 10000,
            use_caching: bool = True,
            chemistry=None,
            **kwargs
    ):
        self.archive = archive
        self.logger = setup_logger(self.__class__.__name__, level=logger_level)
        if chemistry is None:
            chemistry = ChemistryWrapper(use_caching=use_caching)
        self.chemistry = chemistry
        self.step_counter = 0
        self.pb = tqdm if use_tqdm else lambda x: x
        self.log_freq = log_freq
        self.actions = self.action_generator()
        for mol_hash in archive.hashes:
            self._process_new_mol(mol_hash)

    def _process_new_mol(self, mol_hash: int):
        pass

    def process_action(self, action: Action):
        mol_action = self.archive.convert_action_ids_to_mols(action)
        new_mols = self.chemistry.process_action(mol_action)
        for new_mol in new_mols:
            action = action.add_res(hash(new_mol))
            if hash(new_mol) not in self.archive:
                self.archive.add(new_mol, action)
                self._process_new_mol(hash(new_mol))

    @abc.abstractmethod
    def action_generator(self):
        pass

    def search_step(self):
        action = next(self.actions)
        self.step_counter += 1
        self.process_action(action)

    def search(self, n_steps: int, target_mol: Optional[Molecule] = None) -> bool:
        target_mol_hash = None if target_mol is None else hash(target_mol)
        self.logger.debug("Starting search")

        for step in self.pb(range(n_steps)):
            self.search_step()

            if step % self.log_freq == 0:
                self.logger.debug(f"archive length: {len(self.archive)}")
                self.logger.debug(f"state: {dict(self.archive.stats)}")

            if target_mol_hash is not None and target_mol_hash in self.archive:
                return True

            if len(self.archive) >= self.archive.max_len:
                break

        self.logger.debug("Finished search")
        return False


class GraphSearchExhaustive(GraphSearchBase):
    def __init__(
            self,
            archive: MolGraph,
            logger_level: str = "INFO",
            use_tqdm: bool = True,
            chemistry=None,
            **kwargs
    ):
        super().__init__(archive=archive, logger_level=logger_level, use_tqdm=use_tqdm, chemistry=chemistry, **kwargs)
        self.primary_pointer = 0
        self.secondary_pointer = -1

    def _break(self, mol_hash: int):
        mol = self.archive.get(mol_hash)
        for action in self.chemistry.get_valid_actions(mol, op="break"):
            yield action

    def _join(self, mol1_hash: int, mol2_hash: int):
        mol1 = self.archive.get(mol1_hash)
        mol2 = self.archive.get(mol2_hash)

        actions = self.chemistry.get_valid_actions(mol1, mol2, op="join")
        for action in actions:
            yield action

    def action_generator(self):
        while True:
            if self.secondary_pointer == -1:
                mol_hash = self.archive.hashes[self.primary_pointer]
                yield from self._break(mol_hash)
            else:
                mol1_hash = self.archive.hashes[self.primary_pointer]
                mol2_hash = self.archive.hashes[self.secondary_pointer]
                yield from self._join(mol1_hash, mol2_hash)

            if self.secondary_pointer == self.primary_pointer:
                self.primary_pointer += 1
                self.secondary_pointer = -1
            else:
                self.secondary_pointer += 1


class GraphSearchRandom(GraphSearchBase):
    def __init__(
            self,
            archive: MolGraph,
            max_depth: Optional[int] = None,
            seed: int = 0,
            logger_level: str = "INFO",
            use_tqdm: bool = True,
            weight_by_steps: bool = False,
            chemistry=None,
            rng=None,
            **kwargs
    ):
        super().__init__(archive, logger_level, use_tqdm, chemistry=chemistry, **kwargs)
        self.max_depth = max_depth
        self.weight_by_steps = weight_by_steps
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        self.rng = rng
        self.update_freq = 1
        self.n = 0

    def _break(self, mol_hash: int):
        mol = self.archive.get(mol_hash)
        actions = self.chemistry.get_valid_actions(mol, op="break")

        if len(actions) > 0:
            i = self.rng.choice(len(actions))
            action = actions[i]
            yield action

    def _join(self, mol1_hash: int, mol2_hash: int):
        mol1 = self.archive.get(mol1_hash)
        mol2 = self.archive.get(mol2_hash)

        actions = self.chemistry.get_valid_actions(mol1, mol2, op="join")

        if len(actions) > 0:
            i = self.rng.choice(len(actions))
            action = actions[i]
            yield action

    def sample_mols(self, n: Optional[int] = None):
        if self.weight_by_steps:
            self.archive._min_steps.clear()
            weights = [self.archive.get_score(i, cached=True) for i in range(self.n)]
            probs = np.array(weights) / np.sum(weights)
            return self.rng.choice(self.n, n, p=probs)
        else:
            return self.rng.choice(self.n, n)

    def action_generator(self):
        while True:
            if self.step_counter % self.update_freq == 0:
                self.n = len(self.archive)

            op_id = self.rng.choice(2)
            if op_id == 0:
                mol_id = self.sample_mols()
                mol_hash = self.archive.hashes[mol_id]
                if self.max_depth is None or self.archive.get_min_steps(mol_hash) < self.max_depth:
                    yield from self._break(mol_hash)

            elif op_id == 1:
                mol_ids = self.sample_mols(2)
                mol_hashes = [self.archive.hashes[i] for i in mol_ids]
                if self.max_depth is None or [self.archive.get_min_steps(h) < self.max_depth for h in mol_hashes]:
                    yield from self._join(*mol_hashes)
            else:
                raise RuntimeError(f"Unexpected op: {op_id}")


class GraphSearchHeuristic(GraphSearchExhaustive):
    def __init__(
            self,
            archive: MolGraph,
            target_mol: Molecule,
            max_inventory_size: int = 1000,
            logger_level: str = "INFO",
            use_tqdm: bool = True,
            target_match_weight: float = 0.1,
            chemistry=None,
            **kwargs
    ):
        self.inventory = PriorityQueue(max_inventory_size)
        self.target_shifts_matrix = heuristic_tools.construct_shifts_matrix(target_mol.atoms)
        self.target_match_weight = target_match_weight

        super().__init__(
            archive=archive,
            logger_level=logger_level,
            use_tqdm=use_tqdm,
            chemistry=chemistry,
            **kwargs
        )

        self.primary_pointer = 0
        self.secondary_pointer = 0

    def score_mol(self, molecule: Molecule) -> float:
        return heuristic_tools.score_inventory(
            [molecule], self.target_shifts_matrix, self.target_match_weight
        ).item()

    def _process_new_mol(self, mol_hash: int):
        mol = self.archive[mol_hash]
        score = self.score_mol(mol)
        self.inventory.add(mol_hash, score)

    def action_generator(self):
        while True:
            yield from self._join(self.inventory[self.primary_pointer], self.inventory[self.secondary_pointer])
            self.secondary_pointer += 1

            if self.secondary_pointer == len(self.inventory):
                self.secondary_pointer = 0
                self.primary_pointer = (self.primary_pointer + 1) % len(self.inventory)
                yield from self._break(self.inventory[self.primary_pointer])


if __name__ == '__main__':
    archive = MolGraph([Molecule([[1]])])
    target_mol = generate_random_mol(seed=0, n_atoms=10)
    search1 = GraphSearchExhaustive(archive)
    search2 = GraphSearchRandom(archive)
    search3 = GraphSearchHeuristic(archive, target_mol, max_inventory_size=500)
    # for i in tqdm(range(1000)):
    #     search2.search_step()

    success = search3.search(10000, target_mol)
    if success:
        archive.plot_graph(hash(target_mol))
