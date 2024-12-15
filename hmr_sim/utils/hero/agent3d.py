import numpy as np
from hmr_sim.utils.connectivity_controller import ConnectivityController
from hmr_sim.utils.agent import Agent
import math
from collections import defaultdict, deque
from copy import deepcopy

class Agent3D(Agent):
    '''Represents a single independent.'''
    
    def __init__(self, type, agent_id, init_position,
                 dt, vis_radius, map_resolution, 
                 config, controller_params, path_planner=None,
                 path = None, goal = None, init_battery = None, 
                 battery_decay_rate = None, battery_threshold=None,
                 show_old_path=0):
        
        super().__init__(type, agent_id, init_position,
                 dt, vis_radius, map_resolution, 
                 config, controller_params, path_planner,
                 path, goal, init_battery, 
                 battery_decay_rate, battery_threshold,
                 show_old_path)

        if self.type == 'UAV':
            self.state[2] = 1.0
    