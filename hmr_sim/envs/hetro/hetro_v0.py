
import numpy as np
import json
from gymnasium import spaces

from hmr_sim.envs.homo.base import BaseEnv

from hmr_sim.utils.swarm import HeterogeneousSwarm
from hmr_sim.utils.init_formation import init_homo_formation
from hmr_sim.utils.vis import render_homo

class HetroV0(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

        self.num_agents = self.parse_config_entry(config.get('num_agents', []),
                                                  entry_name='num_agents',
                                                  type='array')
        self.total_agents = np.sum(self.num_agents)

        #______________________  Formation Initialization  ______________________
        # Parse initialization positions and formation from the config file
        self.init_positions = self.parse_config_entry(config.get('init_positions', "[]"), "init_positions",type='array')
        self.init_formation = self.parse_config_entry(config.get('init_formation', "[]"), "init_formation")

        print(self.init_positions)
        # Initialize agents based on available configuration
        if self.init_positions.any() and len(self.init_positions) > 0:
            print("Using custom initialization positions.")
            self.agents_positions = self.init_positions
        elif self.init_formation and len(self.init_formation) > 0:
            print(f"Using initialization formation: {self.init_formation}")
            self.positions = []
            for i in range(len(self.init_formation)):
                self.positions.append(init_homo_formation(self.init_formation[i], self.num_agents[i]))   
            self.init_positions = np.vstack(self.positions)

        else:
            raise ValueError("No valid initialization configuration provided.")
        #___________________________________________________________________________

        self.swarm = HeterogeneousSwarm(
            num_agents=self.num_agents,
            init_positions=self.init_positions,
            speed=self.speed,
            dt=self.dt,
            vis_radius=self.vis_radius,
            is_line_of_sight_free_fn=self.is_line_of_sight_free  # Corrected parameter name
        )


        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_agents, 2), dtype=np.float32)

        

    def render(self, mode='human'):
        render_homo(self)

    def parse_config_entry(self, entry, entry_name, type='none'):
        """Parse and validate a configuration entry."""
        try:
            parsed_entry = json.loads(entry)
            return np.array(parsed_entry) if type == "array" else parsed_entry
        except json.JSONDecodeError:
            raise ValueError(f"Invalid format for {entry_name} in config.")
