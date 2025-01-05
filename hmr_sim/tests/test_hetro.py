import random
import sys
from os import path

import gymnasium as gym  # Ensure updated Gymnasium import
import numpy as np
import yaml

# This import has to be left for the script to find the registered environments of hmr_sim
import hmr_sim

from hmr_sim.utils.battery_tracker import BatteryTracker

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

    # create a battery tracker to keep battery states
    battery_tracker = BatteryTracker()

    env.render()

    while not done:
        # action = env.unwrapped.get_dummy_action() #+ [0.5, 0.0]
        # if t < 100:
        # try:
        # if t > 30 and t < 100:
        if t > 30:
            env.unwrapped.controller()
            # if t % 10 == 0:
            battery_tracker.update(env.unwrapped.swarm.agents)
        # obs, reward, terminated, truncated, info = env.step(None)
        # done = terminated or truncated
        # except:
        #     pass
        # Render the environment

        env.render()

        # print(t)

        # Print debug info (optional)
        # if t % 10 == 0:  # Log every 10 steps
        #     print(f"Step {t}: Reward: {reward}, Done: {done}")

        t += 1
        # print(t)
        # if t>6000:
        #     battery_tracker.save_to_file("battery_data.csv")
        #     break


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
