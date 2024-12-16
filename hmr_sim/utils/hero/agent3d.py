import numpy as np
from hmr_sim.utils.connectivity_controller import ConnectivityController
from hmr_sim.utils.agent import Agent
import math
from collections import defaultdict, deque
from copy import deepcopy

class UAV(Agent):
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

        # active, landed, in-transit
        self.mode = 'landed'
        self.hover_height = 2.0

    def take_off(self):
        if self.state[2] < self.hover_height:
            self.state[2] += 0.15
            self.mode = 'in-transit'
        else:
            self.mode = 'active'

    
    def run_controller(self, swarm):
    
        if self.mode == 'landed' or self.mode == 'in-transit':
            self.take_off()

        elif self.controller_type == 'connectivity_controller':
            
            A, id_to_index = self.compute_adjacency()

            v = self.controller(self.agent_id,
                                self.get_pos(),
                                self.neighbors,
                                A,
                                id_to_index)

            proposed_position = self.state[:2] + self.speed * v * self.dt

            # Adjust position using obstacle avoidance
            if self.obstacle_avoidance:
                adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
                                                            is_free_path_fn=swarm.is_line_of_sight_free_fn)
                # Update state with the adjusted position
                self.state[:2] = adjusted_position
            else:
                self.state[:2] = proposed_position

            element = deepcopy(self.state[:2])        
            # self.update_path_history(element)

