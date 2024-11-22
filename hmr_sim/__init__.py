from gymnasium.envs.registration import register

# Register the Homo-v0 environment
register(
    id="Homo-v0",  # Name of the environment
    entry_point="hmr_sim.envs.homo:HomoV0",  # Correct path to the Homo class
    max_episode_steps=1000000000,
)