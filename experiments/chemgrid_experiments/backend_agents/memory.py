import copy
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_backend import Action
from chemgrid_game.game_backend import State
from chemgrid_game.mol_archive import MolArchive


class IDMapper:
    def __init__(self):
        self.oldk2newk = {}
        self.newk2oldk = []

    def map2newk(self, oldk):
        if oldk not in self.oldk2newk:
            self.oldk2newk[oldk] = len(self.oldk2newk)
            self.newk2oldk.append(oldk)

        return self.oldk2newk[oldk]

    def map2oldk(self, new_k):
        return self.newk2oldk[new_k]


class History:
    def __init__(self, n_agents: int, targets: List[Molecule] = None):
        self.n_agents = n_agents
        self.mapper = IDMapper()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []
        self.infos = []
        self.targets = targets

    def add(self, state, action, reward, done, new_state, infos):
        self.states.append(copy.deepcopy(state))
        self.actions.append(copy.deepcopy(action))
        self.rewards.append(copy.deepcopy(reward))
        self.dones.append(copy.deepcopy(done))
        self.new_states.append(copy.deepcopy(new_state))
        self.infos.append(copy.deepcopy(infos))

    def process_state(self, state: State):
        inventory, target, contracts = state
        inventory = [self.mapper.map2newk(mol_id) for mol_id in inventory]
        target = self.mapper.map2newk(hash(target))
        contracts = [(self.mapper.map2newk(c[0]), self.mapper.map2newk(c[1])) for c in contracts]
        return inventory, target, contracts

    def process_action(self, action: Action):
        name, operands, params = action.op, action.operands, action.params
        operands = [self.mapper.map2newk(mol_id) for mol_id in operands]
        return name, operands, params

    def get_data(self, agent_id: int) -> pd.DataFrame:
        states = np.array(self.states, dtype=object)[:, agent_id].tolist()
        states = [self.process_state(s) for s in states]
        actions = np.array(self.actions, dtype=object)[:, agent_id].tolist()
        actions = [self.process_action(a) for a in actions]
        rewards = np.array(self.rewards, dtype=object)[:, agent_id].tolist()
        dones = np.array(self.dones, dtype=object)[:, agent_id].tolist()
        new_states = np.array(self.new_states, dtype=object)[:, agent_id].tolist()
        new_states = [self.process_state(s) for s in new_states]
        infos = np.array(self.infos, dtype=object)[:, agent_id].tolist()

        return pd.DataFrame({
            "state": states,
            "action": actions,
            # "reward": rewards,
            "done": dones,
            "new_state": new_states,
            "info": infos,
        })

    def get_inventory(self, agent_id: int) -> List[int]:
        return np.array(self.new_states, dtype=object)[:, agent_id].tolist()[-1][0]

    def get_build_history(self, agent_id: int, archive: MolArchive = None) -> List[Action]:
        bh = np.array(self.infos, dtype=object)[:, agent_id].tolist()[-1]["build_history"]
        if archive is not None:
            # bh = [(op, [archive[p] for p in ps], archive[c]) for op, ps, c in bh]
            bh = [(Action(op, [archive[p] for p in ps], res=archive[c])) for op, ps, c in bh]
        return bh

    def to_df(self) -> pd.DataFrame:
        dfs = [self.get_data(i) for i in range(self.n_agents)]
        names = [f"agent_{i}" for i in range(self.n_agents)]

        return pd.concat(dfs, axis=1, keys=names)


class ExtendedHistory(History):
    def process_plan(self, plan: List[Action]):
        plan_str = []
        for action in plan:
            mol_id = self.mapper.map2newk(action.res)
            operands = [self.mapper.map2newk(op) for op in action.operands]
            s = f"{action.op} {operands} -> {mol_id}"
            plan_str.append(s)

        return ", ".join(plan_str)

    def process_build_history(self, build_history):
        res = []
        for op_name, parent_hashes, new_mol_hash in build_history:
            mol_id = self.mapper.map2newk(new_mol_hash)
            parent_hashes = [self.mapper.map2newk(op) for op in parent_hashes]
            s = f"{op_name} {parent_hashes} -> {mol_id}"
            res.append(s)

        return ", ".join(res)

    def get_data(self, agent_id: int) -> pd.DataFrame:
        df = super().get_data(agent_id)
        state_df = df["state"].apply(pd.Series)
        new_state_df = df["new_state"].apply(pd.Series)
        state_df.columns = ["inventory", "target", "contracts"]
        new_state_df.columns = ["new_inventory", "new_target", "new_contracts"]
        df = state_df.join(df.drop(columns=["state"]))
        df = df.drop(columns=["new_state"]).join(new_state_df.drop(columns=["new_target"]))
        df = df.drop(columns=["info"]).join(df["info"].apply(pd.Series))
        df["plan_length"] = df["plan"].apply(len)
        df["plan"] = df["plan"].apply(self.process_plan)
        df["build_history"] = df["build_history"].apply(self.process_build_history)

        return df

    def get_history_stats(self):
        df = self.to_df()
        df2 = df.swaplevel(0, 1, 1).stack()

        actions = [op for op, *_ in df2["action"].values]
        c = Counter(actions)
        res = {k: v / self.n_agents for k, v in c.items()}
        res["n_agents"] = self.n_agents
        return res
