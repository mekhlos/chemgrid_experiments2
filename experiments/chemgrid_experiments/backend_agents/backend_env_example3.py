import matplotlib.pyplot as plt
from chemgrid_game.chemistry.utils import create_unit_mol
from chemgrid_game.chemistry.utils import generate_random_mol
from chemgrid_game.gym_env.backend_env import ChemGridBackendEnv

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
env.render()
img = env.render("rgb_array")
plt.imshow(img)
plt.axis("off")
plt.show()
