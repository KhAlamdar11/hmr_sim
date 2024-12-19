from os import path
import sys
import numpy as np
import time
import random
import configparser
import gymnasium as gym  # Ensure updated Gymnasium import
import hmr_sim

import yaml

def run(args):
    # Load environment
    env_name = args.get('env')
    print(f"Initializing environment: {env_name}")

    # Pass the `args` configuration to the environment via kwargs
    env = gym.make(env_name, config=args)

    # Set random seeds for reproducibility
    seed = args.get('seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    # Environment reset
    obs, _ = env.reset()
    done = False
    t = 0

    start_time = time.time()  # Record start time of the iteration
    # elapsed_time = time.time() - start_time  # Measure time taken by the render call

    # if remaining_time > 0:
    #     time.sleep(remaining_time)  # Sleep for the remaining time

    while not done:
        # start_time = time.time()
        env.unwrapped.controller()
        # print(f"Controller time: {time.time()-start_time}")

        # start_time = time.time()
        env.render()
        # print(f"Render time: {time.time()-start_time}")

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

    # Load the configuration
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Run the simulation using the default section of the config
    run(config)

if __name__ == "__main__":
    main()
