from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_backend import Action
from chemgrid_game.game_backend import Contract
from chemgrid_game.game_backend import GameBackend
from chemgrid_game.game_backend import State
from chemgrid_game.mol_archive import MolArchive


class GameBackendWrapper(GameBackend):
    def __init__(
            self,
            inventories: Tuple[List[Molecule]],
            targets: Tuple[Molecule],
            contracts: Set[Contract],
            archive: MolArchive = None,
            logging_level="INFO",
            chemistry=None
    ):
        super().__init__(
            inventories=inventories,
            targets=targets,
            contracts=contracts,
            archive=archive,
            logging_level=logging_level,
            chemistry=chemistry
        )
        self.build_history = [[] for _ in range(self.n_agents)]
        self.reset()

    def add_to_build_history(self, agent_id, item):
        if item not in self.build_history[agent_id]:
            self.build_history[agent_id].append(item)

    def add_mol(
            self,
            agent_id: int,
            mol: Optional[Molecule] = None,
            mol_id: Optional[int] = None,
            parent_op: Optional[str] = None,
            parent_ids: Optional[List[int]] = None
    ):
        if mol_id is None:
            mol_id = hash(mol)
            self.archive[mol_id] = mol
        if mol_id not in self.inventories[agent_id]:
            self.inventories[agent_id].append(mol_id)
            self.contract_queue.append((agent_id, mol_id))
            self.add_to_build_history(agent_id, (parent_op, parent_ids, mol_id))

    def step(self, actions: Tuple[Action], action_infos: List[Dict] = None) -> Tuple:
        infos = action_infos
        [self._step_one(a_id, action) for a_id, action in enumerate(actions)]
        self.check_contracts()

        for i in range(self.n_agents):
            infos[i]["build_history"] = self.build_history[i]

        return self.get_states(), self.compute_rewards(), self.is_done(), infos

    def is_done(self) -> List[bool]:
        return self.get_reached_target()

    def reset(self) -> Tuple[State]:
        states = super().reset()
        for i in range(self.n_agents):
            self.build_history[i].clear()

        return states
