import gymnasium as gym
import numpy as np
from pathlib import Path

import cv2
import yaml

from hmr_sim.controllers.frontier_explore import FrontierDetector

from math import sin, cos

class BaseEnv(gym.Env):

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

        # map setup
        self.map_name = config.get('map_name')
        PACKAGE_ROOT = Path(__file__).resolve().parents[2]
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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
        return self.swarm.get_dummy_action()

    def render(self, mode='human'):
        """Renders the swarm."""
        pass

    def controller(self):
        self.swarm.run_controllers()

    def update_exploration_map(self, state, local_obs, n_angles, sensor):
        current_x, current_y = state
        for i in range(n_angles):
            angle = i * (2 * np.pi / n_angles)
            found_obstacle = False
            if local_obs[i] < sensor:
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


    def get_frontier_goal(self, state):
        self.frontier_detector.set_map(self.exploration_map)
        frontier_map = self.frontier_detector.detect_frontiers()
        candidate_points, labelled_frontiers = self.frontier_detector.label_frontiers(frontier_map)
        ordered_points = self.frontier_detector.nearest_frontier(candidate_points, state)

        if len(ordered_points) > 0:
            goal = np.array(ordered_points[0])
        else:
            print("NO GOAL FOUND")

        return goal