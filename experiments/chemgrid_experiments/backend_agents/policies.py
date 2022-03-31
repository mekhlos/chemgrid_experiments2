from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from chemgrid_game.game_backend import Action
from chemgrid_game.game_backend import State
from chemgrid_game.utils import setup_logger

from experiments.chemgrid_experiments.backend_agents.planner import Planner
from experiments.chemgrid_experiments.graph_search import heuristic_tools


class PolicyBase:
    def __init__(self, planner: Planner, contract_th: int = 1, logger=None, logger_level="INFO"):
        self.planner = planner
        self.archive = self.planner.archive
        self.contract_th = contract_th

        if logger is None:
            logger = setup_logger(self.__class__.__name__, logger_level)

        self.logger = logger
        self.plan_cache = {}

    def get_plan(self, inventory: List[int], target: int):
        k = frozenset(inventory).union([target])
        if k not in self.plan_cache:
            res = self.planner.plan(inventory, target)
            self.plan_cache[k] = res["path"]

        return self.plan_cache[k]

    @abstractmethod
    def compute_one(self, agent_id: int, states: List[State]) -> Tuple[Action, Dict]:
        pass

    def __call__(
            self, state: List[State], include_info: bool = False
    ) -> Union[Tuple[List[Action], List[Tuple]], List[Action]]:

        res = [self.compute_one(a_id, state) for a_id in range(len(state))]
        actions, infos = zip(*res)

        if include_info:
            return actions, infos
        else:
            return actions


class FuturePlanPolicy(PolicyBase):
    def __init__(
            self,
            planner: Planner,
            contract_th: int = 1,
            logger=None,
            logger_level="INFO",
            contract_selection_method: str = "heuristic",
            n_contract_candidates: int = 1
    ):
        super().__init__(planner, contract_th, logger, logger_level)
        self.contract_selection_method = contract_selection_method
        self.n_contract_candidates = n_contract_candidates

    def get_heuristic_scores(self, ids: List[int], target_id: int) -> Tuple[Dict[int, float], List[int]]:
        if len(ids) == 0:
            return {}, []

        mols = [self.archive[i] for i in ids]
        target = self.archive[target_id]
        target_shifts = heuristic_tools.construct_shifts_matrix(target.atoms)
        target_match_weight = self.planner.search_params["target_match_weight"]
        scores = heuristic_tools.score_inventory(mols, target_shifts, target_match_weight)
        scores_dict = dict(zip(ids, scores))
        sorted_ids = sorted(ids, key=lambda x: scores_dict[x], reverse=True)
        return scores_dict, sorted_ids

    def get_cost(self, inventory: List[int], target: int):
        plan = self.get_plan(inventory, target)
        if len(plan) == 0:
            return 0

        if plan[-1].res == target:
            return len(plan)

        return np.inf

    def eval_contract(self, agent_id: int, give: int, get: int, state: List) -> float:
        inventory, target, _ = state[agent_id]
        c1 = self.get_cost(inventory, target)
        c2 = self.get_cost(inventory, give)
        c3 = self.get_cost(inventory + [get], target)

        if np.isfinite(c2) and np.isfinite(c3):
            if np.isfinite(c1):
                return (c1 - c2 - c3 - self.contract_th) / (c1 + 1)
            return 1

        return -1

    def plan_to_inventory(self, plan: List[Action]) -> List[int]:
        return [a.res for a in plan]

    def get_contract_candidates(self, options: List[int], target_id: int):
        if self.contract_selection_method == "random":
            offer_ids = np.random.permutation(options)
        elif self.contract_selection_method == "heuristic":
            offer_scores, offer_ids = self.get_heuristic_scores(options, target_id)
        else:
            raise ValueError(f"Unknown contract selection method: {self.contract_selection_method}")

        return offer_ids[:self.n_contract_candidates]

    def find_contract(self, agent_id: int, plan: List[Action], state: List, n_trials: int = 1):
        inventory, target, contracts = state[agent_id]
        # inventory = self.archive.get_dict(inventory)
        for other_agent_id, (other_inventory, other_target, _) in enumerate(state):
            if other_agent_id == agent_id:
                continue

            other_plan = self.get_plan(other_inventory, other_target)

            extended_inv = set(inventory).union(self.plan_to_inventory(plan)).difference(other_inventory)
            other_extended_inv = set(other_inventory).union(self.plan_to_inventory(other_plan)).difference(inventory)
            offer_ids = self.get_contract_candidates(extended_inv, other_target)
            ask_ids = self.get_contract_candidates(other_extended_inv, target)
            for offer_id in offer_ids[:n_trials]:
                for ask_id in ask_ids[:n_trials]:
                    # self.logger.debug(f"Evaluating contract {agent_id, offer_id, ask_id}")
                    score1 = self.eval_contract(agent_id, offer_id, ask_id, state)
                    score2 = self.eval_contract(other_agent_id, ask_id, offer_id, state)
                    self.logger.debug(f"Evaluating {(offer_id, ask_id)}, scores: {score1}, {score2}")
                    if score1 > 0 and score2 >= 0:
                        return score1, score2, agent_id, other_agent_id, ask_id, offer_id

    def compute_one(self, agent_id: int, state: List[State]) -> Tuple[Action, Dict]:
        self.logger.debug(f"Agent {agent_id}'s turn")

        inventory, target, contracts = state[agent_id]

        cs = [c for c in contracts if c[0] == agent_id]
        for contract in cs:
            score = self.eval_contract(*contract, state=state)
            if score > 0:
                plan = self.get_plan(inventory, contract[1])
                if len(plan) > 0:
                    self.logger.debug(f"Acting on contract created by agent: {contract}")
                    return plan[0], {"plan": plan}

        for other_aid, offer, ask in contracts:
            score = self.eval_contract(agent_id, ask, offer, state)
            if score > 0:
                plan = self.get_plan(inventory, ask)
                if len(plan) > 0:
                    self.logger.debug(f"Acting on contract created by another agent: {(other_aid, offer, ask)}")
                    return plan[0], {"plan": plan}

        plan = self.get_plan(inventory, target)

        contract = self.find_contract(agent_id, plan, state)
        if contract is None:
            if len(plan) > 0:
                self.logger.debug(f"Failed to offer new contract, executing first step of plan: {plan}")
                return plan[0], {"plan": plan}

            self.logger.debug(f"Failed to generate plan, sending noop")
            return Action(), {"plan": plan}

        n_saved_steps, n_saved_steps_for_other, agent_id, other_agent_id, mol_to_offer, mol_to_ask = contract
        if (agent_id, mol_to_offer, mol_to_ask) in contracts:
            plan = self.get_plan(inventory, mol_to_offer)
            if len(plan) > 0:
                self.logger.debug(f"Contract exists, acting on it {contract}")
                return plan[0], {"plan": plan}

        self.logger.debug(f"Creating new contract: {contract}")
        action = Action("contract", (mol_to_offer, mol_to_ask))
        info = {
            "plan": plan,
            "saved_steps": n_saved_steps,
            "saved_steps_for_other": n_saved_steps_for_other
        }
        return action, info
