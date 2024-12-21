
import numpy as np
import json
from gymnasium import spaces
import ast

from hmr_sim.envs.hetro.base import BaseEnv

from hmr_sim.utils.swarm import Swarm
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.vis import SwarmRenderer

np.random.seed(12)

class HetroV0(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

        self.start = True

        self.agent_config = config.get('agent_config')

        self.vis_radius = config.get('vis_radius')

        self.total_agents = sum(inner_dict["num_agents"] for inner_dict in self.agent_config.values())
        self.old_total_agents = self.total_agents

        self.render_type = config.get('vis_params')['render_type']
        self.show_old_path = config.get('vis_params')['show_old_path']


        self.swarm = Swarm(env=self,
                           config = config,
                           map_resolution=self.resolution,
                           map_handlers={'update_exploration_map': self.update_exploration_map,
                                        'is_free_space': self.is_free_space,
                                        'is_line_of_sight_free': self.is_line_of_sight_free,
                                        'get_frontier_goal': self.get_frontier_goal})

        self.render_func = SwarmRenderer(render_type=self.render_type,
                                         env=self,
                                         swarm=self.swarm, occupancy_grid=self.occupancy_grid, 
                                        origin=self.origin, resolution=self.resolution, 
                                        vis_radius=self.vis_radius, 
                                        plot_limits = config.get('vis_params')['plot_limits'] if config.get('vis_params')['plot_limits'] != 'None' else None)
            
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_agents, 4), dtype=np.float64)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_agents, 2), dtype=np.float32)

        

    def render(self, mode='human'):

        # if self.old_total_agents > self.swarm.total_agents:
        #     if self.swarm.total_agents % 2 == 0: 
        self.render_func.render()
        #         self.old_total_agents = self.swarm.total_agents
        # elif self.start:
        #     self.render_func.render()
        #     self.start = False

