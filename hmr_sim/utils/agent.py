import numpy as np

class Agent:
    '''Represents a single independent.'''
    
    def __init__(self, agent_id, init_pos, speed, dt, vis_radius, type=0):
        self.agent_id = agent_id
        self.state = np.zeros(4)
        self.state[:2] = init_pos
        self.speed = speed
        self.dt = dt
        self.vis_radius = vis_radius
        self.path = []
        self.type = type

    
    def update_state(self, action, is_free_space_fn):
        velocity = action * self.speed
        proposed_position = self.state[:2] + velocity * self.dt

        if is_free_space_fn(proposed_position):
            self.state[2:] = velocity
            self.state[:2] = proposed_position  
        else:
            self.state[2:] = 0 
            print(f"Agent {self.agent_id}: Invalid action, position blocked.")

        
    def compute_local_observation(self, occupancy_grid, origin, resolution):
        '''Computes local sensor observations.'''
        pass


    def follow_path(self):
        '''Makes the agent follow the path waypoint by waypoint.'''
        pass