from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from chemgrid_game.chemistry.mol_chemistry import ChemistryWrapper
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.chemistry.utils import create_unit_mol
from chemgrid_game.chemistry.utils import generate_random_mol
from chemgrid_game.mol_archive import MolArchive
from chemgrid_game.utils import setup_logger

from experiments.chemgrid_experiments.backend_agents.game_backend_wrapper import GameBackendWrapper
from experiments.chemgrid_experiments.backend_agents.memory import ExtendedHistory
from experiments.chemgrid_experiments.backend_agents.planner import Planner
from experiments.chemgrid_experiments.backend_agents.policies import FuturePlanPolicy
from experiments.chemgrid_experiments.backend_agents.policies import PolicyBase
from experiments.chemgrid_experiments.graph_search.search import GraphSearchBase
from experiments.chemgrid_experiments.graph_search.search import GraphSearchHeuristic
from experiments.chemgrid_experiments.graph_search.search import GraphSearchRandom
from experiments.chemgrid_experiments.mol_graph import MolGraph


def plan_to_seq(plan: List[Dict], mol_dict, as_mols=True) -> List:
    f = lambda x: mol_dict[x] if as_mols else x

    res = []
    for step in plan:
        c = step["mol_hash"]
        op, ps, _ = step["action"]
        res.append((op, [f(p) for p in ps], f(c)))

    return res


def generate_random_mols_via_search(
        n_colors: int = 1,
        n_steps: int = 1000,
        seed: int = 0,
        mol_size: Optional[int] = None,
        initial_mols: Optional[List[Molecule]] = None
) -> List[Molecule]:
    if initial_mols is None:
        initial_mols = [create_unit_mol(i + 1) for i in range(n_colors)]
    archive = MolGraph(initial_mols=initial_mols)
    gs = GraphSearchRandom(archive, max_depth=10, seed=seed)
    gs.search(n_steps=n_steps)
    mols = [mol for mol in archive.get_molecules() if mol_size is None or mol.atoms.sum() == mol_size]
    return mols


def generate_random_mols(n: int, mol_size: int, rng=None, n_colors: int = 1, grid_size=8) -> List[Molecule]:
    if rng is None:
        rng = np.random.default_rng(seed=0)
    seeds = rng.integers(0, 1000, n)
    mols = [generate_random_mol(s, mol_size, n_colors, grid_size=grid_size) for s in seeds]
    return mols


def generate_initial_inventory(n_colors: int):
    return [create_unit_mol(i + 1, grid_size=target_mols[0].grid_size) for i in range(n_colors)]


class ExperimentRunner:

    def __init__(
            self,
            planner_class: Type[Planner] = Planner,
            search_class: Type[GraphSearchBase] = GraphSearchHeuristic,
            search_params: Dict = None,
            policy_class: Type[PolicyBase] = FuturePlanPolicy,
            policy_params: Dict = None,
            env_class: Type[GameBackendWrapper] = GameBackendWrapper,
            max_steps: int = 100,
            target_mols: Optional[List[Molecule]] = None,
            inventories: Optional[Union[List, Molecule]] = None,
            mol_size: Optional[int] = None,
            rng: Optional = None,
            n_agents: Optional[int] = None,
            logger=None,
            logger_level: str = "INFO",
            chemistry=None
    ):
        self.max_steps = max_steps
        if logger is None:
            logger = setup_logger("experiment", logger_level)

        if target_mols is None:
            logger.debug("Generating random mols...")
            target_mols = generate_random_mols(n_agents, mol_size, rng)
            logger.debug("Done.")

        n_agents = len(target_mols)

        if inventories is None:
            logger.debug("Generating inventories...")
            inventories = [generate_initial_inventory(max(tm.atoms)) for tm in target_mols]
            logger.debug("Done.")
        elif isinstance(inventories, Molecule):
            logger.debug("Sharing initial inventory for all agents...")
            inventories = [[inventories] for _ in range(n_agents)]
            logger.debug("Done.")
        # elif isinstance(inventories, List):
        #     logger.debug("Generating inventory dict from inventory list")
        #     inventories = [[mol for mol in inventory] for inventory in inventories]
        #     logger.debug("Done.")

        if chemistry is None:
            chemistry = ChemistryWrapper()

        contracts = set()
        archive = MolArchive([i for inv in inventories for i in inv] + target_mols)
        planner = planner_class(archive, search_class, **search_params)
        self.policy = policy_class(planner, **policy_params)
        self.env = env_class(tuple(inventories), tuple(target_mols), contracts, archive=archive, chemistry=chemistry)
        self.history = ExtendedHistory(n_agents, targets=target_mols)

    def run(self):
        state = self.env.reset()
        for i in range(self.max_steps):
            actions, action_infos = self.policy(state, include_info=True)
            new_state, rewards, dones, infos = self.env.step(actions, action_infos=action_infos)
            self.history.add(state, actions, rewards, dones, new_state, infos)

            if all(dones):
                break

            state = new_state

        return self.history


if __name__ == '__main__':
    mol_size = 10
    rng = np.random.default_rng(seed=0)
    n_agents = 5

    target_mols = generate_random_mols(n_agents, mol_size, rng)

    planner_params = dict(
        search_type="heuristic4",
        max_steps=100000,
        max_inventory_size=1000,
        use_tqdm=False,
        target_match_weight=0.1
    )
    policy_params = dict(contract_th=1)

    experiment = ExperimentRunner(target_mols=target_mols, planner_params=planner_params, policy_params=policy_params)
    history = experiment.run()
