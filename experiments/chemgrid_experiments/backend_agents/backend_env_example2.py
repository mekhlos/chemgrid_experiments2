import numpy as np
from chemgrid_game.chemistry.utils import create_unit_mol
from chemgrid_game.chemistry.utils import generate_random_mol
from chemgrid_game.gym_env.backend_env import ChemGridBackendEnv
from chemgrid_game.plotting import plot_atoms_list

inventory = [create_unit_mol(1, grid_size=4)]
target = generate_random_mol(0, 5, grid_size=4)
env = ChemGridBackendEnv(
    initial_inventories=[inventory],
    initial_targets=[target],
    grid_size=4,
    logging_level="DEBUG"
)
print(env.observation_space.sample().shape)
print(env.action_space.sample().shape)
print(env.reset().shape)

actions = [
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [2, 0, 0],
    [2, 0, 0],
    [0, 0, 2],
    [3, 0, 0],
    [0, 0, 0],
]

state = env.reset()
plot_atoms_list(state, background=True, m=1, n=len(state))

for action in actions:
    new_state, reward, done, info = env.step(np.array(action))
    plot_atoms_list(new_state, background=True, m=1, n=len(new_state))
