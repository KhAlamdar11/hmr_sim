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

        #________________________  controller  ________________________
        if controller_params is not None:
            self.controller = ConnectivityController(params=controller_params)

    
    def update_state(self, swarm, action, is_free_space_fn):
        velocity = action * self.speed
        proposed_position = self.state[:2] + velocity * self.dt

        adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
                                                        is_free_path_fn=swarm.is_line_of_sight_free_fn)

        # Update state with the adjusted position
        self.state[:2] = adjusted_position

    def set_path(self,path):
        self.path = path
        self.state[:2] = path[0]
        self.path_len = len(self.path)

        
    def compute_local_observation(self, occupancy_grid, origin, resolution):
        '''Computes local sensor observations.'''
        pass


    def run_controller(self, swarm):
        """
        Runs the controller and integrates obstacle avoidance.

        Parameters:
        swarm (Swarm): The swarm object containing other agents.
        poses (np.ndarray): Current poses of all agents.
        sensor_radius (float): Radius within which the agent checks for obstacles.
        is_free_path_fn (function): Function to check if the path is free of obstacles.
        """
        if self.path is not None:
            self.state[:2] = self.path[self.path_idx % self.path_len]
            self.path_idx += 1
        else:
            # Run connectivity controller to compute the velocity
            v = self.controller(self.agent_id,
                                self.get_pos(),
                                self.neighbors,
                                swarm.compute_adjacency_matrix())
            
            # Compute proposed position
            proposed_position = self.state[:2] + self.speed * v * self.dt

            # Adjust position using obstacle avoidance
            adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
                                                        is_free_path_fn=swarm.is_line_of_sight_free_fn)

            # Update state with the adjusted position
            self.state[:2] = adjusted_position



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
    
    def obstacle_avoidance(self, proposed_position, is_free_path_fn, num_samples=16,sensor_radius=0.5):
        """
        Adjust the proposed position to avoid collisions using free path sampling.

        Parameters:
        proposed_position (np.ndarray): The next position proposed by the controller.
        sensor_radius (float): The radius within which the agent checks for obstacles.
        is_free_path_fn (function): Function to check if the path between two points is free of obstacles.
        num_samples (int): Number of directions to sample around the agent.

        Returns:
        np.ndarray: Adjusted position to avoid collisions.
        """
        current_position = self.state[:2]
        direction_to_target = proposed_position - current_position
        magnitude = np.linalg.norm(direction_to_target)
        direction_to_target /= np.linalg.norm(direction_to_target)  # Normalize

        check_point = current_position + direction_to_target * sensor_radius

        # print(direction_to_target)
        # print(check_point)

        # Check if the direct path to the proposed position is free
        if is_free_path_fn(current_position, check_point):
            return proposed_position
        
        # Sample alternative directions
        best_direction = None
        max_clear_distance = 0

        # Generate alternating positive and negative angles
        base_angles = np.linspace(0, np.pi, num_samples // 2, endpoint=False)
        angles = np.empty((num_samples,))
        angles[0::2] = base_angles  # Fill even indices with positive angles
        angles[1::2] = -base_angles  # Fill odd indices with negative angles

        for angle in angles:
            # Generate a candidate direction
            candidate_direction = np.array([
                np.cos(angle) * sensor_radius,
                np.sin(angle) * sensor_radius
            ])
            candidate_position = current_position + candidate_direction

            # Check if the candidate path is free
            if is_free_path_fn(current_position, candidate_position):
                clear_distance = np.linalg.norm(candidate_direction)
                if clear_distance > max_clear_distance:
                    max_clear_distance = clear_distance
                    best_direction = candidate_direction

        # If a best direction is found, move in that direction
        if best_direction is not None:

            # normalize the best direction
            best_direction /= np.linalg.norm(best_direction) 
            best_direction *= magnitude

            adjusted_position = current_position + best_direction
            return adjusted_position

        # If no direction is free, stay in the current position
        print(f"Agent {self.agent_id}: No clear path, staying in place.")
        return current_position
