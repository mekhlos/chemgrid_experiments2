from pathlib import Path
from typing import List
from typing import Set
from typing import Tuple

import graphviz
import numpy as np
from chemgrid_game.chemistry.molecule import Molecule
from chemgrid_game.game_backend import Action
from chemgrid_game.utils import get_datetime_str

from experiments.files import FILES_DIR

Edge = Tuple[str, Molecule, Molecule]


class GraphVizGraph:
    def __init__(self):
        self.graph = graphviz.Digraph()
        self.nodes = set()

    def add_node(self, node_id, **kwargs):
        node_id = str(node_id)
        self.nodes.add(node_id)
        self.graph.node(node_id, **kwargs)

    def add_edge(self, parent, child, **kwargs):
        parent, child = str(parent), str(child)
        self.graph.edge(parent, child, **kwargs)

    def get_nodes(self) -> Set:
        return self.nodes

    def save(self, path=None, suffix: str = "") -> str:
        if path is None:
            path = FILES_DIR.joinpath("graphs")
        else:
            path = Path(path)

        t = get_datetime_str()
        p = path.joinpath(f"graph_{t}{suffix}")
        self.graph.render(p, format="png")
        p.unlink()
        return p.with_suffix(".png")

    def __contains__(self, item):
        return item in self.nodes


def save_graph_to_img(graph: GraphVizGraph, dir_path=FILES_DIR.joinpath("graphs")) -> str:
    print(f"Saving graph to {dir_path}")
    s = "".join(chr(i) for i in np.random.choice(ord("z") - ord("a"), 3) + ord("a"))
    if not dir_path.is_dir():
        dir_path.mkdir(exist_ok=True, parents=True)
    return graph.save(path=dir_path, suffix=s)


def seq_to_graph(seq: List[Action]):
    edges = seq_to_edges(seq)
    return edge_list_to_graph(edges)


def seq_to_edges(seq: List[Action]) -> List[Edge]:
    edges = []
    for i, action in enumerate(seq):
        for parent in action.operands:
            edges.append((action.op, parent, action.res))

    return edges


def edge_list_to_graph(edges: List[Edge]):
    graph = GraphVizGraph()

    def add_node(mol: Molecule):
        mol_hash = hash(mol)
        if mol_hash not in graph:
            if not Path(mol.get_img_path()).is_file():
                mol.render(save_fig=True, core_only=True)
            graph.add_node(mol_hash, image=str(mol.get_img_path()), shape="none", label="")

    def add_edge(op: str, parent: Molecule, child: Molecule):
        if op == "join":
            style, color = "solid", "black"
        elif op == "break":
            style, color = "dashed", "purple"
        elif op == "contract":
            style, color = "dotted", "brown"
        else:
            raise ValueError(f"Unknown op '{op}'")

        mol_hash = hash(child)
        parent_hash = hash(parent)
        graph.add_edge(parent_hash, mol_hash, color=color, label="", style=style, penwidth="8")

    for i, (op, parent, child) in enumerate(edges):
        add_node(child)
        add_node(parent)
        add_edge(op, parent, child)

    return graph
