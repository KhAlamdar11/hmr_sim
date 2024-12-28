import math
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np

from hmr_sim.utils.connectivity_controller import ConnectivityController

"""
Represents an individual agent in the swarm.

This class models an agent with its state, behavior, and interactions in a 
multi-agent swarm simulation. It supports various controller types and includes 
functionality for obstacle avoidance, path following, and goal-directed behavior.

Attributes:
    type (str): The type of the agent.
    agent_id (int): Unique identifier for the agent.
    state (np.ndarray): The state of the agent [x, y, vx, vy].
    battery (float): Battery level of the agent.
    neighbors (list): List of neighboring agents.
"""


class Agent:
    def __init__(self, type, agent_id, init_position,
                 dt, vis_radius, map_resolution,
                 config, controller_params, path_planner=None,
                 path=None, goal=None, init_battery=None,
                 battery_decay_rate=None, battery_threshold=None,
                 show_old_path=0):

        # Base parameters
        self.type = type
        self.agent_id = agent_id
        self.dt = dt
        self.vis_radius = vis_radius
        self.map_resolution = map_resolution
        self.mode = 'active'

        # Configuration parameters
        self.is_obstacle_avoidance = config['obstacle_avoidance']
        self.speed = config['speed']
        self.sensor_radius = config['sensor_radius']
        self.obstacle_radius = config['obs_radius']

        # State variables
        self.state = np.zeros(4) # [x, y, vx, vy]
        self.state[:2] = init_position
        self.path = None
        self.path_idx = 0
        self.path_len = 0
        self.neighbors = None

        # Battery variables
        self.battery = init_battery if init_battery is not None else 1.0
        self.battery_decay_rate = battery_decay_rate
        self.battery_threshold = battery_threshold

        # Debugging and edge-case handling
        self.n_agents = None
        self.prev_n_agents = None
        self.problem = False

        # Path history visualization
        self.old_path_len = show_old_path
        self.old_path = []

        # Controller setup
        self.controller_type = config['controller_type']
        print(f'Agent ID: {self.agent_id}, Type: {self.type}, Controller: {self.controller_type}')

        if self.controller_type == 'connectivity_controller':
            self.controller = ConnectivityController(params=controller_params)
        elif self.controller_type == 'go_to_goal':
            print(f"Agent ID: {self.agent_id} -> Planning path to {goal}")
            self.controller = path_planner
            self.controller.set_goal(goal)
            self.set_path(path_planner.plan_path(self.state[:2]))
            print(f"Agent ID: {self.agent_id} -> Path set to {goal}")
        elif self.controller_type == 'explore':
            self.controller = path_planner
        elif self.controller_type == 'path_tracker':
            self.set_path(path)

    def set_path(self, path):
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

            # Adjust position using obstacle avoidance
            if self.is_obstacle_avoidance:
                adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position,
                                                            is_free_path_fn=swarm.is_line_of_sight_free_fn)
                # Update state with the adjusted position
                self.state[:2] = adjusted_position
            else:
                self.state[:2] = proposed_position
            element = deepcopy(self.state[:2])

            # Update path history for visualization
            self.update_path_history(element)

        # Path tracker: Follow a predefined path
        elif self.controller_type == 'path_tracker' and self.path is not None:
            self.state[:2] = self.path[self.path_idx % self.path_len]
            self.path_idx += 1

        # Goal-directed controller: Move towards a goal
        elif self.controller_type == 'go_to_goal' and self.path is not None:
            if self.path_idx < self.path_len:
                displacement = self.path[self.path_idx] - self.state[:2]
                if np.linalg.norm(displacement) > 0.05:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed if np.linalg.norm(
                        displacement) != 0 else np.zeros_like(displacement)
                else:
                    self.state[:2] = self.path[self.path_idx]
                    self.path_idx += 1

            # self.prev_n_agents = deepcopy(self.n_agents)

            # _, self.n_agents = self.compute_adjacency()
            # self.n_agents = len(self.n_agents.keys())

            # if not(self.problem):
            #     if self.prev_n_agents is not None and self.n_agents is not None:
            #         if self.n_agents < self.prev_n_agents - 1:
            #             self.path = self.path[:self.path_idx]
            #             self.path = self.path[::-1]
            #             self.path_idx = 0
            #             self.problem = True

        # Exploration: Update map and plan new paths based on frontier exploration
        elif self.controller_type == 'explore':
            local_obs, n_angles = self.get_local_observation(swarm.is_free_space_fn)
            swarm.update_exploration_map_fn(self.state[:2], local_obs, n_angles, self.sensor_radius)

            if self.path is None:
                goal = swarm.get_frontier_goal_fn(self.state[:2])
                print("GOAL: ", goal)
                self.controller.set_goal(goal)
                self.set_path(self.controller.plan_path(self.state[:2]))

            if self.path_idx < self.path_len:
                displacement = self.path[self.path_idx] - self.state[:2]
                if np.linalg.norm(displacement) > 0.05:
                    self.state[:2] += (displacement / np.linalg.norm(displacement)) * self.speed if np.linalg.norm(
                        displacement) != 0 else np.zeros_like(displacement)
                else:
                    self.state[:2] = self.path[self.path_idx]
                    self.path_idx += 1
            else:
                print("path reached")
                self.path = None

        elif self.controller_type == 'dummy':
            self.state[0] += 1 * self.speed

        elif self.controller_type == 'do_not_move':
            pass

        # Battery decay logic
        if self.battery_decay_rate is not None and self.battery > 0:
            self.battery = max(0, self.battery - self.battery_decay_rate)


    def get_local_observation(self, is_free_space_fn, n_angles=360):
        """
        Simulates a local observation using a simulated LIDAR-like sensor. It calculates the distances to obstacles in the agent's surroundings
        by casting rays in `n_angles` directions up to the agent's sensor radius.

        Args:
            is_free_space_fn (callable): A function that checks if a given position in space is free.
            n_angles (int, optional): Number of angles to sample in a 360-degree field of view. Defaults to 360.

        Returns:
            tuple:
                - local_obs (numpy.ndarray): Array of distances to obstacles for each sampled angle.
                - n_angles (int): Number of angles sampled (same as input).
        """
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

    def compute_adjacency(self):
        """
        Computes the adjacency matrix representing the communication graph of the swarmby using BFS to traverse the swarm's agents and build an adjacency matrix.

        Returns:
            tuple:
                - adjacency_matrix (numpy.ndarray): A binary matrix where a value of 1
                  indicates a direct connection between agents.
                - id_to_index (dict): A mapping from agent IDs to their respective
                  indices in the adjacency matrix.
        """

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


    def obstacle_avoidance(self, proposed_position, is_free_path_fn, num_samples=4):
        """
        Adjust the proposed position to avoid collisions using free path sampling by steer to avoid methodology.

        Parameters:
        proposed_position (np.ndarray): The next position proposed by the controller.
        obstacle_radius (float): The radius within which the agent checks for obstacles.
        is_free_path_fn (function): Function to check if the path between two points is free of obstacles.
        num_samples (int): Number of directions to sample around the agent.

        Returns:
        np.ndarray: Adjusted position to avoid collisions.
        """
        current_position = self.state[:2]
        direction_to_target = proposed_position - current_position
        magnitude = np.linalg.norm(direction_to_target)
        direction_to_target /= np.linalg.norm(direction_to_target)  # Normalize

        check_point = current_position + direction_to_target * self.obstacle_radius

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
                np.cos(angle) * self.obstacle_radius,
                np.sin(angle) * self.obstacle_radius
            ])
            candidate_position = current_position + candidate_direction

            # Check if the candidate path is free
            if is_free_path_fn(current_position, candidate_position):
                clear_distance = np.linalg.norm(candidate_direction)
                if clear_distance > max_clear_distance:
                    max_clear_distance = clear_distance
                    best_direction = candidate_direction

        # If the best direction is found, move in that direction
        if best_direction is not None:
            # normalize the best direction
            best_direction /= np.linalg.norm(best_direction)
            best_direction *= magnitude

            adjusted_position = current_position + best_direction
            return adjusted_position

        # If no direction is free, stay in the current position
        print(f"Agent {self.agent_id}: No clear path, staying in place.")
        return current_position

    def get_id(self):
        return self.agent_id

    def set_neighbors(self, neighbors):
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
        sorted_paired = sorted(paired)  # Sort by ID (ascending)
        sorted_positions = [pos for _, pos in sorted_paired]  # Extract positions

        return np.array(sorted_positions)

    def get_pos(self):
        return self.state[:2]

    def set_pos(self, pos):
        print(f"Type: {self.type}, ID: {self.agent_id}, battery type: {self.battery}")
        self.state[:2] = pos

    def is_battery_critical(self):
        if self.battery_decay_rate is not None:
            if self.battery <= self.battery_threshold:
                return True
        return False

    def update_path_history(self, element):
        if len(self.old_path) >= self.old_path_len:
            self.old_path.pop(0)  # Remove the oldest element

        if len(self.old_path) > 0:
            distance = np.linalg.norm(element - self.old_path[-1])
            if distance > 0.1:
                self.old_path.append(element)  # Add the new element
        elif len(self.old_path) == 0:
            self.old_path.append(element)  # Add the new element

    def get_data(self):
        return {"id": self.agent_id, "position": self.state[:2]}

    # def update_state(self, swarm, action, is_free_space_fn):
    #     velocity = action * self.speed
    #     proposed_position = self.state[:2] + velocity * self.dt

    #     adjusted_position = self.obstacle_avoidance(proposed_position=proposed_position,
    #                                                     is_free_path_fn=swarm.is_line_of_sight_free_fn)

    #     # Update state with the adjusted position
    #     self.state[:2] = adjusted_position
