import numpy as np
from hmr_sim.utils.connectivity_controller import ConnectivityController
import math
from collections import defaultdict, deque
from copy import deepcopy

class Agent:
    '''Represents a single independent.'''
    
    def __init__(self, agent_id, init_pos, speed, dt, vis_radius, 
                 type, controller_type, path_planner, 
                 controller_params, map_resolution, obstacle_avoidance):
        
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

        self.is_obstacle_avoidance = obstacle_avoidance

        self.n_agents = None 
        self.prev_n_agents = None    
        self.problem = False

        self.map_resolution = map_resolution

        self.obstacle_radius = controller_params['obs_radius']
        self.sensor_radius = controller_params['sensor_radius']

        #________________________  controller  ________________________
        self.controller_type = controller_type

        if self.controller_type == 'connectivity_controller':
            self.controller = ConnectivityController(params=controller_params)
        elif self.controller_type == 'go_to_goal' and path_planner is not None:
            print("planning path")
            self.controller = path_planner
            self.set_path(path_planner.plan_path(self.state[:2]))
        elif self.controller_type == 'explore':
            self.controller = path_planner


    # def update_state(self, swarm, action, is_free_space_fn):
    #     velocity = action * self.speed
    #     proposed_position = self.state[:2] + velocity * self.dt

    #     adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
    #                                                     is_free_path_fn=swarm.is_line_of_sight_free_fn)

    #     # Update state with the adjusted position
    #     self.state[:2] = adjusted_position

    def set_path(self,path):
        self.path = path
        self.state[:2] = path[0]
        self.path_len = len(self.path)
        self.path_idx = 0


    def run_controller(self, swarm):
        """
        Runs the controller and integrates obstacle avoidance.

        Parameters:
        swarm (Swarm): The swarm object containing other agents.
        """

        if self.controller_type == 'connectivity_controller':
            
            A, id_to_index = self.compute_adjacency()

            v = self.controller(self.agent_id,
                                self.get_pos(),
                                self.neighbors,
                                A,
                                id_to_index)

            proposed_position = self.state[:2] + self.speed * v * self.dt

            print("Prop", proposed_position)

            # Adjust position using obstacle avoidance
            print("OBS", self.is_obstacle_avoidance)
            if self.is_obstacle_avoidance:
                print("YES")
                adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position, 
                                                            is_free_path_fn=swarm.is_line_of_sight_free_fn)
                # Update state with the adjusted position
                self.state[:2] = adjusted_position
            else:
                print("NO")
                self.state[:2] = proposed_position

            print("state", self.state[:2])

        
        elif self.controller_type == 'path_tracker' and self.path is not None:
            self.state[:2] = self.path[self.path_idx % self.path_len]
            self.path_idx += 1

        elif self.controller_type == 'go_to_goal' and self.path is not None:
            if self.path_idx < self.path_len:
                displacement = self.path[self.path_idx] - self.state[:2]
                if np.linalg.norm(displacement) > 0.05:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed if np.linalg.norm(displacement) != 0 else np.zeros_like(displacement)
                else:
                    self.state[:2] = self.path[self.path_idx]
                    self.path_idx+=1

            self.prev_n_agents = deepcopy(self.n_agents)

            _, self.n_agents = self.compute_adjacency()
            self.n_agents = len(self.n_agents.keys())

            if not(self.problem):
                if self.prev_n_agents is not None and self.n_agents is not None:
                    if self.n_agents < self.prev_n_agents - 1:
                        self.path = self.path[:self.path_idx]
                        self.path = self.path[::-1]
                        self.path_idx = 0
                        self.problem = True

        elif self.controller_type == 'explore':
            local_obs,n_angles = self.get_local_observation(swarm.is_free_space_fn)
            swarm.update_exploration_map_fn(self.state[:2],local_obs,n_angles,self.sensor_radius)
            
            if self.path is None:
                goal = swarm.get_frontier_goal_fn(self.state[:2])
                print("GOAL: ", goal)
                self.controller.set_goal(goal)
                self.set_path(self.controller.plan_path(self.state[:2]))

            if self.path_idx < self.path_len:
                displacement = self.path[self.path_idx] - self.state[:2]
                if np.linalg.norm(displacement) > 0.05:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed if np.linalg.norm(displacement) != 0 else np.zeros_like(displacement)
                else:
                    self.state[:2] = self.path[self.path_idx]
                    self.path_idx+=1
            else:
                print("path reached")
                self.path = None
        elif self.controller_type == 'dummy':
            self.state[0] += 1*self.speed



    def get_local_observation(self, is_free_space_fn, n_angles=360):
        local_obs = np.full(n_angles, self.sensor_radius)
        current_x, current_y = self.state[:2]
        for i in range(n_angles):
            angle = i * (2 * np.pi / n_angles)
            for r in np.linspace(0, self.sensor_radius, int(self.sensor_radius / self.map_resolution)):
                x = current_x + r * math.cos(angle)
                y = current_y + r * math.sin(angle)
                if not is_free_space_fn((x, y)):
                    local_obs[i] = r
                    break
        return local_obs, n_angles



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

        return np.array(matrix), id_to_index


    def get_id(self):
        return self.agent_id


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
    
    # def obstacle_avoidance(self, proposed_position, is_free_path_fn, num_samples=4):
    #     """
    #     Adjust the proposed position to avoid collisions using free path sampling.

    #     Parameters:
    #     proposed_position (np.ndarray): The next position proposed by the controller.
    #     obstacle_radius (float): The radius within which the agent checks for obstacles.
    #     is_free_path_fn (function): Function to check if the path between two points is free of obstacles.
    #     num_samples (int): Number of directions to sample around the agent.

    #     Returns:
    #     np.ndarray: Adjusted position to avoid collisions.
    #     """
    #     current_position = self.state[:2]
    #     direction_to_target = proposed_position - current_position
    #     magnitude = np.linalg.norm(direction_to_target)
    #     direction_to_target /= np.linalg.norm(direction_to_target)  # Normalize

    #     check_point = current_position + direction_to_target * self.obstacle_radius

    #     # print(direction_to_target)
    #     # print(check_point)

    #     # Check if the direct path to the proposed position is free
    #     if is_free_path_fn(current_position, check_point):
    #         return proposed_position
        
    #     # Sample alternative directions
    #     best_direction = None
    #     max_clear_distance = 0

    #     # Generate alternating positive and negative angles
    #     base_angles = np.linspace(0, np.pi, num_samples // 2, endpoint=False)
    #     angles = np.empty((num_samples,))
    #     angles[0::2] = base_angles  # Fill even indices with positive angles
    #     angles[1::2] = -base_angles  # Fill odd indices with negative angles

    #     for angle in angles:
    #         # Generate a candidate direction
    #         candidate_direction = np.array([
    #             np.cos(angle) * self.obstacle_radius,
    #             np.sin(angle) * self.obstacle_radius
    #         ])
    #         candidate_position = current_position + candidate_direction

    #         # Check if the candidate path is free
    #         if is_free_path_fn(current_position, candidate_position):
    #             clear_distance = np.linalg.norm(candidate_direction)
    #             if clear_distance > max_clear_distance:
    #                 max_clear_distance = clear_distance
    #                 best_direction = candidate_direction

    #     # If a best direction is found, move in that direction
    #     if best_direction is not None:

    #         # normalize the best direction
    #         best_direction /= np.linalg.norm(best_direction) 
    #         best_direction *= magnitude

    #         adjusted_position = current_position + best_direction
    #         return adjusted_position

    #     # If no direction is free, stay in the current position
    #     print(f"Agent {self.agent_id}: No clear path, staying in place.")
    #     return current_position
    
