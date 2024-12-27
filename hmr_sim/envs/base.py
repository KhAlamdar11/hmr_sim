from math import sin, cos
from pathlib import Path

import cv2
from gymnasium import Env
import numpy as np
import yaml

from hmr_sim.controllers.frontier_explore import FrontierDetector


class BaseEnv(Env):
    def __init__(self, config):
        super().__init__()

        self.dt = config.get('dt')  # Returns float

        self.occupancy_grid = None  # To be loaded via config
        self.origin = None
        self.resolution = None

        self.fig = None

        # exploration map variables
        self.exploration_map = None
        self.frontier_detector = None

        # swarm variables
        self.swarm = None

        # map setup
        self.map_name = config.get('map_name')
        PACKAGE_ROOT = Path(__file__).resolve().parents[1]
        self.maps_folder = PACKAGE_ROOT / "maps" / self.map_name
        self.image_path = self.maps_folder / "map.bmp"
        self.yaml_path = self.maps_folder / "data.yaml"
        self.occupancy_grid = self.load_occupancy_grid(str(self.image_path))
        self.origin, self.resolution = self.load_yaml_config(str(self.yaml_path))

        self.exploration_map = np.full_like(self.occupancy_grid, -1)
        self.frontier_detector = FrontierDetector(self.exploration_map, self.resolution,
                                                  [self.origin['x'], self.origin['y']], robot_size=0.5)

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        origin = {
            'x': float(config.get('origin', {}).get('x', 0.0)),
            'y': float(config.get('origin', {}).get('y', 0.0))
        }
        resolution = float(config.get('resolution', 0.1))
        return origin, resolution

    def load_occupancy_grid(self, image_path):
        import os

        # Debugging: Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File does not exist: {image_path}")

        # Attempt to read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                f"Failed to load image from {image_path}. Ensure the file exists and is a valid BMP image.")

        # Convert the image to a binary grid
        _, binary_grid = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        return 1.0 - binary_grid

    def position_to_grid(self, position):
        grid_x = int((position[0] - self.origin['x']) / self.resolution)
        grid_y = int((position[1] - self.origin['y']) / self.resolution)
        return grid_x, grid_y

    def is_free_space(self, position):
        grid_x, grid_y = self.position_to_grid(position)
        if 0 <= grid_x < self.occupancy_grid.shape[1] and 0 <= grid_y < self.occupancy_grid.shape[0]:
            return self.occupancy_grid[grid_y, grid_x] == 0
        return True

    # TODO: Refactor this function
    def is_line_of_sight_free(self, position1, position2):
        """
        Checks if the line of sight between two positions is obstacle-free.

        Args:
            position1 (np.ndarray): [x, y] of the first position.
            position2 (np.ndarray): [x, y] of the second position.

        Returns:
            bool: True if line of sight is free, False otherwise.
        """

        def position_to_grid(position):
            grid_x = int((position[0] - self.origin['x']) / self.resolution)
            grid_y = int((position[1] - self.origin['y']) / self.resolution)
            return grid_x, grid_y

        start = position_to_grid(position1)
        end = position_to_grid(position2)

        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Check if the current grid cell is an obstacle
            if self.occupancy_grid[y0, x0] == 1:
                return False

            if (x0, y0) == (x1, y1):
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # for i, agent in enumerate(self.swarm.agents):
        #     if agent.path is None:
        #         agent.state[:2] = self.init_positions[i]
        #         agent.state[2:] = 0
        return self.swarm.get_states(), {}

    def step(self, actions=None):
        if actions is not None:
            self.swarm.step(actions, self.is_free_space)

        rewards = np.zeros(self.num_agents)
        terminated = False
        truncated = False
        obs = self.swarm.get_states()
        return obs, rewards, terminated, truncated, {}

    def get_dummy_action(self):
        """
        Creates a dummy action as defined in the swarm class. Usually a translation.

        Returns:
            numpy.ndarray: An n x k array representing actions for all states.
        """
        return self.swarm.get_dummy_action()

    def render(self, mode='human'):
        pass

    # TODO: return all the velocities from the controller for learning purposes
    def controller(self):
        self.swarm.run_controllers()

    def update_exploration_map(self, state, local_obs, n_angles, sensor_radius):
        """
        Updates the exploration map given

        Args:
            state (numpy.ndarray): position of the agent
            local_obs (list): lidar values
            n_angles (int): resolution of lidar
            sensor_radius (float): max range of the lidar sensor
        """
        current_x, current_y = state
        for i in range(n_angles):
            angle = i * (2 * np.pi / n_angles)
            found_obstacle = False
            if local_obs[i] < sensor_radius:
                for r in np.linspace(0, local_obs[i], int(local_obs[i] / self.resolution)):
                    x = current_x + r * cos(angle)
                    y = current_y + r * sin(angle)
                    grid_x, grid_y = self.position_to_grid((x, y))
                    if 0 <= grid_x < self.exploration_map.shape[1] and 0 <= grid_y < self.exploration_map.shape[0]:
                        if not self.is_free_space((x, y)):
                            self.exploration_map[grid_y, grid_x] = 1
                            found_obstacle = True
                            break
                        else:
                            self.exploration_map[grid_y, grid_x] = 0

    # TODO: include error handling if no goal is present
    def get_frontier_goal(self, state):
        """
        Runs the frontier exploration algorithm to find the closest frontier to the agent's state.

        Args:
            state (numpy.ndarray): The position of the agent.

        Returns:
            numpy.ndarray or None: The frontier goal selected from the exploration map,
        """
        if not isinstance(state, np.ndarray) or state.shape[0] < 2:
            raise ValueError(f"Invalid state provided: {state}. State must be a numpy array with at least 2 elements.")

        try:
            self.frontier_detector.set_map(self.exploration_map)
            frontier_map = self.frontier_detector.detect_frontiers()
            candidate_points, labelled_frontiers = self.frontier_detector.label_frontiers(frontier_map)
            ordered_points = self.frontier_detector.nearest_frontier(candidate_points, state)
        except Exception as e:
            raise RuntimeError(f"Error during frontier detection: {e}")


        if len(ordered_points) > 0:
            goal = np.array(ordered_points[0])
        else:
            print("NO GOAL FOUND")
            goal = None

        return goal
