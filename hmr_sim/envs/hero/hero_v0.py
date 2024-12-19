
import numpy as np
import json
from gymnasium import spaces
import ast
import time
from hmr_sim.envs.hetro.base import BaseEnv

from hmr_sim.utils.hero.swarm3d import Swarm3D
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.hero.vis3d import SwarmRenderer3D

np.random.seed(12)

class HeroV0(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

        self.num_agents = config.get('num_agents', [])

        self.agent_config = config.get('agent_config')

        self.vis_radius = config.get('vis_radius')

        self.total_agents = sum(inner_dict["num_agents"] for inner_dict in self.agent_config.values())

        self.render_type = config.get('vis_params')['render_type']
        self.show_old_path = config.get('vis_params')['show_old_path']

        self.swarm = Swarm3D(env=self,
                           config = config,
                           map_resolution=self.resolution,
                           map_handlers={'update_exploration_map': self.update_exploration_map,
                                        'is_free_space': self.is_free_space,
                                        'is_line_of_sight_free': self.is_line_of_sight_free,
                                        'get_frontier_goal': self.get_frontier_goal})

        self.render_func = SwarmRenderer3D(swarm=self.swarm,occupancy_grid={'map':self.occupancy_grid, 
                                                                            'origin': self.origin, 
                                                                            'res': self.resolution})
            
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_agents, 2), dtype=np.float32)

        

    def render(self, mode='human'):
        self.render_func.render()
        start_time = time.time()
        # self.render_func.render()
        elapsed_time = time.time() - start_time
        remaining_time = self.dt - elapsed_time

        if remaining_time > 0:
            time.sleep(remaining_time)

        # print(f"REMAINUING TIME {remaining_time}")

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
