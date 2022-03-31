from typing import List

import numpy as np
from chemgrid_game.chemistry.molecule import Molecule


def separate_color_dims(x):
    return (x[:, :, :, None].repeat(4, -1) == np.arange(1, 5).reshape((1, 1, 1, 4))).astype(np.uint8)


def construct_shifts_matrix(target: np.ndarray) -> np.ndarray:
    m, n = target.shape
    res = []
    for i in range(2 * m):
        for j in range(2 * n):
            base = np.zeros((3 * m, 3 * n))
            base[i:i + m, j:j + n] = target
            res.append(base)

    x = np.array(res)
    x = separate_color_dims(x)
    return x


def _process_inventory_for_scoring(inventory: List[Molecule]) -> np.ndarray:
    m, n = inventory[0].atoms.shape
    inventory = np.array([m.atoms for m in inventory])
    inventory = np.pad(inventory, ((0, 0), (m, m), (n, n)))
    inventory = separate_color_dims(inventory)
    return inventory


def _score_processed_inventory(inventory: np.ndarray, targets: np.ndarray, target_match_weight) -> np.ndarray:
    n_atoms = inventory.sum(axis=(1, 2, 3)).reshape(1, -1)
    n_target_atoms = targets.sum(axis=(1, 2, 3)).reshape(-1, 1)
    matches = np.einsum("mxyc,nxyc->mn", targets, inventory)
    scores = matches / n_atoms + target_match_weight * (matches / n_target_atoms)
    scores = scores.max(axis=0)
    return scores


def score_inventory(inventory: List[Molecule], target_shifts_matrix: np.ndarray, target_match_weight) -> np.ndarray:
    inventory = _process_inventory_for_scoring(inventory)
    return _score_processed_inventory(inventory, target_shifts_matrix, target_match_weight)
