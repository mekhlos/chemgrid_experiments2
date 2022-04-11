from typing import List
from typing import Optional
from typing import Tuple

from chemgrid_game.chemistry.mol_chemistry import Action

from experiments.chemgrid_experiments.new_chemistry.mols import Molecule


class ChemistryWrapper:
    def __init__(self, use_caching=True):
        self.use_caching = use_caching
        self.bond_combos = [(c1, c2) for c1 in range(1, 4) for c2 in range(1, 4)]

    def _get_all_joins(self, mol1: Molecule, mol2: Molecule) -> List[Molecule]:
        res = []
        for bond in self.bond_combos:
            res.extend(self.join_mols(mol1, mol2, bond, symmetric=False))

        return res

    def _get_all_breaks(self, mol: Molecule) -> List[Molecule]:
        res = []
        for bond in self.bond_combos:
            res.extend(self.break_mol(mol, bond))

        return res

    def join_mols(self, mol1: Molecule, mol2: Molecule, bond: Tuple[int, int], symmetric=True) -> List[Molecule]:
        if bond is None:
            return self._get_all_joins(mol1, mol2)
        _, joins = mol1.do_join(mol2, bond)
        if symmetric:
            _, more_joins = mol2.do_join(mol1, bond)
            for mol in more_joins:
                if mol not in joins:
                    joins.append(mol)
        return joins

    def break_mol(self, mol: Molecule, bond: Tuple[int, int] = None) -> List[Molecule]:
        if bond is None:
            return self._get_all_breaks(mol)
        _, breaks = mol.do_break(bond)
        return breaks

    def get_valid_actions(
            self,
            mol1: Molecule,
            mol2: Optional[Molecule] = None,
            op: Optional[str] = None,
            check_join_valid: bool = False
    ):
        actions = []
        if op == "break" or op is None:
            for bond in self.bond_combos:
                actions.append(Action(op="break", operands=(hash(mol1),), params=bond))

        if mol2 is not None and (op == "join" or op is None):
            for bond in self.bond_combos:
                hash1, hash2 = hash(mol1), hash(mol2)
                actions.append(Action(op="join", operands=(hash1, hash2), params=bond))

        return actions

    def process_action(self, action: Action, check_valid=True) -> List[Molecule]:
        if action.op == "join":
            return self.join_mols(*action.operands, bond=action.params)
        elif action.op == "break":
            return self.break_mol(*action.operands, bond=action.params)
        else:
            raise ValueError(f"Unknown op {action.op}")
