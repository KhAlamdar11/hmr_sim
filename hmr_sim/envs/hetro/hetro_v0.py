
import numpy as np
import json
from gymnasium import spaces
import ast

from hmr_sim.envs.homo.base import BaseEnv

from hmr_sim.utils.swarm import HeterogeneousSwarm
from hmr_sim.utils.utils import get_curve
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

        # Initialize agents based on available configuration
        if self.init_positions.any() and len(self.init_positions) > 0:
            print("Using custom initialization positions.")
            self.agents_positions = self.init_positions
        elif self.init_formation and len(self.init_formation) > 0:
            print(f"Using initialization formation: {self.init_formation}")
            self.positions = []
            for i in range(len(self.init_formation)):
                self.positions.append(get_curve(self.init_formation[i], self.num_agents[i]))   
            self.init_positions = np.vstack(self.positions)

        else:
            raise ValueError("No valid initialization configuration provided.")
        #___________________________________________________________________________       

        self.speed = self.parse_config_entry(config.get('robot_speed', fallback="{}"), "robot_speed", type='dict')

        self.vis_radius = config.getfloat('vis_radius', 5.0)  # Returns float

        self.swarm = HeterogeneousSwarm(
            num_agents=self.num_agents,
            init_positions=self.init_positions,
            speed=self.speed,
            dt=self.dt,
            vis_radius=self.vis_radius,
            is_line_of_sight_free_fn=self.is_line_of_sight_free, 
            config = config 
        )

        # Paths
        self.paths = self.parse_config_entry(config.get('paths', fallback="{}"), "paths", type='dict')
        self.swarm.set_all_paths(self.paths)            
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_agents, 2), dtype=np.float32)

        

    def render(self, mode='human'):
        render_homo(self)


    def parse_config_entry(self, entry, entry_name, type='none'):
        """Parse and validate a configuration entry."""
        try:
            # Safely evaluate Python-style literals like lists or dictionaries
            parsed_entry = ast.literal_eval(entry)
            if type == "array":
                return np.array(parsed_entry)
            elif type == "dict":
                if not isinstance(parsed_entry, dict):
                    raise ValueError(f"{entry_name} must be a dictionary.")
                # Optionally convert keys to integers if needed
                return {int(k): v for k, v in parsed_entry.items()}
            return parsed_entry
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid format for {entry_name} in config.")
