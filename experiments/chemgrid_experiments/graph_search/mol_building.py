from typing import List
from typing import Tuple

from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_backend import Action
from chemgrid_game.game_backend import Contract


def process_action(
        action: Action,
        chemistry,
        check_valid=True
) -> Tuple[List[Molecule], List[Contract]]:
    new_mols, new_contracts = [], []

    if action.op == "contract":
        offer, ask = action.operands
        new_contracts = (action.params, offer, ask)

    else:
        new_mols = chemistry.process_action(action, check_valid)

    return new_mols, new_contracts
