import numpy as np
from chemgrid_game.chemistry.utils import create_unit_mol
from chemgrid_game.chemistry.utils import generate_random_mol
from chemgrid_game.gym_env.backend_env import ChemGridBackendEnv
from chemgrid_game.plotting import plot_atoms_list
from tqdm import tqdm

inventory = [create_unit_mol(1, grid_size=4)]
target = generate_random_mol(0, 5, grid_size=4)
env = ChemGridBackendEnv(
    initial_inventories=[inventory],
    initial_targets=[target],
    grid_size=4,
    max_inv_size=5,
    logging_level="INFO"
)
print(env.observation_space.sample().shape)
print(env.action_space.sample().shape)
print(env.reset().shape)

state = env.reset()
plot_atoms_list(state, background=True, m=1, n=len(state))

for i in tqdm(range(1000)):
    action = env.action_space.sample()
    new_state, reward, done, info = env.step(np.array(action))
    state = new_state

plot_atoms_list(state, background=True, m=1, n=len(state))
