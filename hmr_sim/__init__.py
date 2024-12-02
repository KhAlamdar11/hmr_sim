from gymnasium.envs.registration import register

register(
    id="Hetro-v0",  # Name of the environment
    entry_point="hmr_sim.envs.hetro:HetroV0",  # Correct path to the Homo class
    max_episode_steps=10000000000000000000000,
)