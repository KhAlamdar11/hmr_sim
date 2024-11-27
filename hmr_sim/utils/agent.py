import numpy as np
from hmr_sim.utils.connectivity_controller import ConnectivityController
import math
from collections import defaultdict, deque


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


    def run_controller(self, swarm):
        """
        Runs the controller and integrates obstacle avoidance.

        Parameters:
        swarm (Swarm): The swarm object containing other agents.
        """
        if self.path is not None:
            self.state[:2] = self.path[self.path_idx % self.path_len]
            self.path_idx += 1
        else:
            # Run connectivity controller to compute the velocity

            A, neighbor_ids = self.compute_adjacency()

            v = self.controller(self.agent_id,
                                self.get_pos(),
                                self.neighbors,
                                A)
            
            # Compute proposed position
            proposed_position = self.state[:2] + self.speed * v * self.dt

            # Adjust position using obstacle avoidance
            adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
                                                        is_free_path_fn=swarm.is_line_of_sight_free_fn)

            # Update state with the adjusted position
            self.state[:2] = adjusted_position


    def get_data(self):
        return {"id": self.agent_id, "position": self.state[:2]}
    
    
    def compute_adjacency(self):

        # Adjacency representation (dynamic growing dictionary)
        adjacency = defaultdict(set)

        # Queue for BFS traversal
        queue = deque([self])
        visited = set()  # Track visited agents by ID

        while queue:
            current_agent = queue.popleft()
            current_info = current_agent.get_data()
            current_id = current_info["id"]

            if current_id in visited:
                continue

            visited.add(current_id)

            # Process current agent's neighbors
            for neighbor in current_agent.neighbors:
                neighbor_info = neighbor.get_data()
                neighbor_id = neighbor_info["id"]

                # Update adjacency structure
                adjacency[current_id].add(neighbor_id)
                adjacency[neighbor_id].add(current_id)  # Ensure it's bidirectional

                if neighbor_id not in visited:
                    queue.append(neighbor)

        # Convert adjacency dictionary to matrix representation
        all_ids = sorted(adjacency.keys())  # Ensure consistent ordering
        id_to_index = {agent_id: index for index, agent_id in enumerate(all_ids)}
        size = len(all_ids)
        matrix = [[0] * size for _ in range(size)]


        for agent_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                i, j = id_to_index[agent_id], id_to_index[neighbor_id]
                matrix[i][j] = 1

        return np.array(matrix), all_ids


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
        ids = []
        
        # Collect IDs and positions from neighbors
        for agent in self.neighbors:
            ids.append(agent.get_id())
            positions.append(agent.get_pos())
        
        # Pair IDs with positions, sort pairs by ID, and extract sorted positions
        paired = list(zip(ids, positions))  # [(id1, pos1), (id2, pos2), ...]
        sorted_paired = sorted(paired)      # Sort by ID (ascending)
        sorted_positions = [pos for _, pos in sorted_paired]  # Extract positions
        
        return np.array(sorted_positions)

    
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
