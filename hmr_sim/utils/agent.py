import numpy as np
from hmr_sim.utils.connectivity_controller import ConnectivityController
import math

class Agent:
    '''Represents a single independent.'''
    
    def __init__(self, agent_id, init_pos, speed, dt, vis_radius, type=0, controller_params=None):
        self.agent_id = agent_id
        self.state = np.zeros(4)
        self.state[:2] = init_pos
        self.speed = speed
        self.dt = dt
        self.vis_radius = vis_radius
        self.path = None
        self.path_idx = 0
        self.path_len = 0
        self.type = type
        self.neighbors = None
        
        # if controller is not None:
        #________________________  controller  ________________________
        # delta = 0.2
        # repelThreshold = 0.6
        # controller_params = {'battery_aware': 0,
        #                             'sigma': math.sqrt(-self.vis_radius/(2*math.log(delta))),
        #                             'range': self.vis_radius,
        #                             'normalized': 0,
        #                             'epsilon': 0.01,
        #                             'gainConnectivity': 1.0,
        #                             'gainRepel': 0.1,
        #                             'repelThreshold': self.vis_radius*repelThreshold,
        #                             'unweighted': 1,
        #                             'v_max': 0.6,
        #                             'critical_battery_level': 0.14,
        #                             'tau': 0.01}
        self.controller = ConnectivityController(params=controller_params)

    
    def update_state(self, action, is_free_space_fn):
        velocity = action * self.speed
        proposed_position = self.state[:2] + velocity * self.dt

        if is_free_space_fn(proposed_position):
            self.state[2:] = velocity
            self.state[:2] = proposed_position  
        else:
            self.state[2:] = 0 
            print(f"Agent {self.agent_id}: Invalid action, position blocked.")


    def set_path(self,path):
        self.path = path
        self.state[:2] = path[0]
        self.path_len = len(self.path)

        
    def compute_local_observation(self, occupancy_grid, origin, resolution):
        '''Computes local sensor observations.'''
        pass


    def run_controller(self,swarm,poses):
        if self.path is not None:
            self.state[:2] = self.path[self.path_idx%self.path_len]
            self.path_idx += 1
        # else:
        #     v = self.controller(self.agent_id,
        #                         poses,
        #                         swarm.compute_adjacency_matrix())
        #     self.state[:2] += self.speed*v*0.7
        else:
            v = self.controller(self.agent_id,
                                self.get_pos(),
                                self.neighbors,
                                swarm.compute_adjacency_matrix())
            self.state[:2] += self.speed*v*0.7


    def get_id(self):
        return self.agent_id


    def follow_path(self):
        '''Makes the agent follow the path waypoint by waypoint.'''
        pass


    def set_neighbors(self,neighbors):
        self.neighbors = neighbors

    
    def get_neighbors(self):
        return self.neighbors
    

    def get_neighbors_pos(self):
        positions = []
        for agent in self.neighbors:
            positions.append(agent.get_pos())
        return np.array(positions)

    
    def get_pos(self):
        return self.state[:2]