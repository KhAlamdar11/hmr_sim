
import numpy as np
import json
from gymnasium import spaces

from hmr_sim.envs.homo.base import BaseEnv

from hmr_sim.utils.swarm import Swarm
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.vis import render_homo

class HomoV0(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

        self.vis_radius = config.getfloat('vis_radius', 5.0)  # Returns float
        self.speed = config.getfloat('robot_speed', 1.0)  # Use getfloat() to ensure speed is a float

        self.num_agents = config.getint('num_agents', 2) 

        #______________________  Formation Initialization  ______________________
        # Parse initialization positions and formation from the config file
        self.init_positions = self.parse_config_entry(config.get('init_positions', "[]"), "init_positions")
        self.init_formation = self.parse_config_entry(config.get('init_formation', "[]"), "init_formation")

        print(self.init_positions)
        # Initialize agents based on available configuration
        if self.init_positions.any() and len(self.init_positions) > 0:
            print("Using custom initialization positions.")
            self.agents_positions = self.init_positions
        elif self.init_formation and len(self.init_formation) > 0:
            print(f"Using initialization formation: {self.init_formation}")
            self.init_positions = get_curve(self.init_formation, self.num_agents)
        else:
            raise ValueError("No valid initialization configuration provided.")
        #___________________________________________________________________________
        
        self.swarm = Swarm(
            num_agents=self.num_agents,
            init_positions=self.init_positions,
            speed=self.speed,
            dt=self.dt,
            vis_radius=self.vis_radius,
            is_line_of_sight_free_fn=self.is_line_of_sight_free  # Corrected parameter name
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents, 2), dtype=np.float32)

    def render(self, mode='human'):
        render_homo(self)

    def parse_config_entry(self, entry, entry_name):
        """Parse and validate a configuration entry."""
        try:
            parsed_entry = json.loads(entry)
            return np.array(parsed_entry) if entry_name == "init_positions" else parsed_entry
        except json.JSONDecodeError:
            raise ValueError(f"Invalid format for '{entry_name}'. Must be a JSON-serializable list.")
