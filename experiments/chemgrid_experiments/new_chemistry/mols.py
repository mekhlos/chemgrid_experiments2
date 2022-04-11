"""
based on the idea and code of Nicholas Guttenberg
"""
from pathlib import Path

import numpy as np
from chemgrid_game import CHEMGRID_GAME_PATH
from chemgrid_game.plotting import plot_atoms
from matplotlib import pyplot as plt


class Molecule:
    def __init__(self, grid_size):
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.parents = []
        self.generation = 0
        self.grid_size = grid_size

    @property
    def atoms(self):
        return self.grid

    def canonicalize(self):
        xmin = np.where(np.sum(self.grid, axis=1) > 0)[0][0]
        ymin = np.where(np.sum(self.grid, axis=0) > 0)[0][0]
        xmax = np.where(np.sum(self.grid, axis=1) > 0)[0][-1]
        ymax = np.where(np.sum(self.grid, axis=0) > 0)[0][-1]

        grid = self.grid[xmin:xmax + 1, ymin:ymax + 1]
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[:xmax - xmin + 1, :ymax - ymin + 1] = grid

    def get_connected_component(self, x0, y0, exclude_bond=None):
        connect = np.zeros((self.grid_size, self.grid_size))

        connect[x0, y0] = 1

        stop = False
        while not stop:
            stop = True
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if connect[x, y]:
                        for d in range(4):
                            if d == 0:
                                xm = x
                                ym = y - 1
                            elif d == 1:
                                xm = x + 1
                                ym = y
                            elif d == 2:
                                xm = x
                                ym = y + 1
                            else:
                                xm = x - 1
                                ym = y

                            if (xm >= 0) and (ym >= 0) and (xm < self.grid_size) and (ym < self.grid_size):
                                if self.grid[xm, ym]:
                                    valid = True
                                    if exclude_bond is not None:
                                        if self.grid[xm, ym] == exclude_bond[0] and self.grid[x, y] == exclude_bond[1]:
                                            valid = False
                                        if self.grid[xm, ym] == exclude_bond[1] and self.grid[x, y] == exclude_bond[0]:
                                            valid = False

                                    if not connect[xm, ym] and valid:
                                        connect[xm, ym] = 1
                                        stop = False
        return connect

    def check_valid(self):
        if np.sum(self.grid > 0) < 2:
            return False

        atoms_x, atoms_y = np.where(self.grid > 0)

        connect = self.get_connected_component(atoms_x[0], atoms_y[0], None)

        for i in range(len(atoms_x)):
            if connect[atoms_x[i], atoms_y[i]] == 0:
                return False

        return True

    def do_break(self, bond):
        atoms_x, atoms_y = np.where(self.grid > 0)

        mol = Molecule(self.grid_size)
        mol.grid = self.grid.copy()

        outputs = []

        for i in range(len(atoms_x)):
            if mol.grid[atoms_x[i], atoms_y[i]]:
                sg = mol.get_connected_component(atoms_x[i], atoms_y[i], bond)

                mol2 = Molecule(self.grid_size)
                mol2.grid = mol.grid * (sg > 0)
                mol2.canonicalize()
                if not mol2.check_valid():
                    return False, []
                mol2.parents = [self.get_hash()]
                mol2.generation = 1 + self.generation
                outputs.append(mol2)
                mol.grid *= (sg == 0)

        if len(outputs) <= 1:
            return False, []

        return True, outputs

    def do_join(self, other, bond):
        max_size = self.grid_size
        supergrid = np.zeros((3 * max_size, 3 * max_size))
        supergrid[max_size:2 * max_size, max_size:2 * max_size] = self.grid
        source = np.zeros((3 * max_size, 3 * max_size))
        source[max_size:2 * max_size, max_size:2 * max_size] = (self.grid > 0) * 1

        origx, origy = np.where(self.grid == bond[0])
        otherx, othery = np.where(other.grid == bond[1])

        for i in range(len(origx)):
            for j in range(len(otherx)):
                for d in range(4):
                    if d == 0:
                        dy = -1
                        dx = 0
                    elif d == 1:
                        dx = 1
                        dy = 0
                    elif d == 2:
                        dx = 0
                        dy = 1
                    elif d == 3:
                        dx = -1
                        dy = 0

                    ofsx = dx + origx[i] - otherx[j]
                    ofsy = dy + origy[i] - othery[j]
                    region = source[ofsx + max_size:ofsx + 2 * max_size,
                             ofsy + max_size:ofsy + 2 * max_size]

                    if np.sum((region == 1) * (other.grid > 0)) == 0:  # Valid
                        if np.sum((region == 2) * (other.grid > 0)) != 0:  # Collision
                            return False, []
                        else:
                            source[ofsx + max_size:ofsx + 2 * max_size,
                            ofsy + max_size:ofsy + 2 * max_size] += 2 * (other.grid > 0)
                            supergrid[ofsx + max_size:ofsx + 2 * max_size,
                            ofsy + max_size:ofsy + 2 * max_size] += other.grid
        if np.sum(source == 2) == 0:  # No join
            return False, []
        xmin = np.where(np.sum(supergrid, axis=1) > 0)[0][0]
        ymin = np.where(np.sum(supergrid, axis=0) > 0)[0][0]
        xmax = np.where(np.sum(supergrid, axis=1) > 0)[0][-1]
        ymax = np.where(np.sum(supergrid, axis=0) > 0)[0][-1]

        if xmax - xmin >= max_size:
            return False, []
        if ymax - ymin >= max_size:
            return False, []

        mol = Molecule(max_size)
        mol.grid[:xmax - xmin + 1, :ymax - ymin + 1] = supergrid[xmin:xmax + 1, ymin:ymax + 1]
        mol.parents = [self.get_hash(), other.get_hash()]
        mol.generation = 1 + self.generation
        if 1 + other.generation > mol.generation:
            mol.generation = 1 + other.generation

        return True, [mol]

    def get_hash(self):
        hashstr = ""
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                hashstr += chr(ord('0') + self.grid[i, j])

        return hashstr

    def __hash__(self):
        return int(self.get_hash())

    def recalculate_generation(self, mol_db):
        self.generation = 0

        for p in self.parents:
            if 1 + mol_db[p.get_hash()].generation > self.generation:
                self.generation = 1 + mol_db[p.get_hash()].generation

    def get_img_path(self) -> Path:
        p = CHEMGRID_GAME_PATH.joinpath(f"files/new_mol_imgs/{hash(self)}.png")
        if not p.parent.is_dir():
            p.parent.mkdir(exist_ok=True)
        return Path(p).resolve()

    def get_mol_path(self) -> Path:
        p = CHEMGRID_GAME_PATH.joinpath(f"files/new_mols/{hash(self)}.json")
        if not p.parent.is_dir():
            p.parent.mkdir(exist_ok=True)
        return p

    def render(self, square_size=1, ax=None, core_only=False, save_fig=False, add_title=False):
        if core_only:
            data = self.atoms
        else:
            data = self.atoms
        fig = None
        if ax is None:
            n_rows, n_bits = data.shape
            h, w = n_rows * square_size, n_bits * square_size
            fig, ax = plt.subplots(figsize=(w, h))

        plot_atoms(data, scale=square_size, ax=ax)

        if add_title:
            ax.set_title(f"{self}")
        if save_fig and fig is not None and not self.get_img_path().is_file():
            fig.savefig(self.get_img_path())

        if fig is not None:
            plt.close()


def try_add(mol, molecules):
    mh = mol.get_hash()

    if mh in molecules:
        if molecules[mh].generation > mol.generation:
            molecules[mh] = mol
    else:
        molecules[mh] = mol


def recursive_render(graph, key, molecules):
    graph.node(key, label="", image="mol_ims/%s.png" % key, shape="circle")
    for p in molecules[key].parents:
        recursive_render(graph, p, molecules)
        graph.edge(p, key)
