from os import path
import sys
import numpy as np
import random
import configparser
import gymnasium as gym  # Ensure updated Gymnasium import
import hmr_sim

def run(args):
    # Load environment
    env_name = args.get('env')
    print(f"Initializing environment: {env_name}")

    # Pass the `args` configuration to the environment via kwargs
    env = gym.make(env_name, config=args)

    # Set random seeds for reproducibility
    seed = args.getint('seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    # Environment reset
    obs, _ = env.reset()
    done = False
    t = 0

    while not done:
        # action = env.unwrapped.get_dummy_action() #+ [0.5, 0.0]
        # if t < 100:
        if t > 30:
            env.unwrapped.controller()
        obs, reward, terminated, truncated, info = env.step(None)
        # done = terminated or truncated

        # Render the environment
        env.render()

        # Print debug info (optional)
        # if t % 10 == 0:  # Log every 10 steps
        #     print(f"Step {t}: Reward: {reward}, Done: {done}")

        t+=1

    print("Simulation ended.")


def main():
    """
    Entry point of the script. Reads configuration and runs the environment.
    """
    # Ensure the configuration file is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python test_script.py <config_file>")
        return

    # Load configuration file
    config_file = sys.argv[1]
    config_file_path = path.join(path.dirname(__file__), config_file)
    print(f"Loading configuration from: {config_file_path}")

    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Run the simulation using the default section of the config
    run(config[config.default_section])

if __name__ == "__main__":
    main()
