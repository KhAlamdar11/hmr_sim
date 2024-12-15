
from hmr_sim.utils.hero.agent3d import Agent3D

# from hmr_sim.utils.agent import Agent as Agent3D

import numpy as np
from scipy.spatial.distance import euclidean
import math
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.rrt import RRT
from hmr_sim.utils.add_agents import AddAgent
from hmr_sim.utils.lattice_generation import gen_lattice
from hmr_sim.utils.swarm import Swarm

class Swarm3D(Swarm):
    """Manages a swarm of heterogeneous agents."""

    def __init__(self, env, config, map_resolution, map_handlers):
        super().__init__(env, config, map_resolution, map_handlers)

        #________________________  instance definitions  ________________________
        self.agents = []
        goals = None
        path_planner = None
        paths = []
        for agent_type in self.agent_config.keys():

            #__________________  Initial position handling  ___________________
            init_position = self.agent_config[agent_type]['init_position']
            if init_position == 'None':
                init_formation = self.agent_config[agent_type]['init_formation']
                print(f"Using initialization formation: {init_formation}")
                if init_formation['shape'] != 'lattice':
                    init_position = get_curve(init_formation, 
                                            self.agent_config[agent_type]['num_agents'])   
                else:
                    start = np.array([0.0,0.0])
                    end = np.array([3.5,0.0])
                    init_position = gen_lattice(self.agent_config[agent_type]['num_agents'], 
                                                self.vis_radius, 
                                                start, end)
            
            goals = None
            path_planner = None
            paths = []


            if self.agent_config[agent_type]['controller_type'] == 'explore':
                path_planner = RRT(env)
            elif self.agent_config[agent_type]['controller_type'] == 'go_to_goal':
                path_planner = RRT(env)
                goals = self.agent_config[agent_type]['goals']
            elif self.agent_config[agent_type]['controller_type'] == 'path_tracker':
                for path in list(self.agent_config[agent_type]['paths'].values()):
                    paths.append(get_curve(path, 
                                            speed=self.agent_config[agent_type]['speed'],
                                            dt=self.dt))

                    # print(f'Agent ID: {self.agent_id}, Type: {self.type}, Controller: {self.controller_type}')
        
            
            # baterries
            init_battery = self.agent_config.get(agent_type).get('init_battery', None)
            if init_battery=='autofill':
                init_battery = np.linspace(0.2, 0.9, self.agent_config[agent_type]['num_agents'])        
            battery_decay_rate = self.agent_config.get(agent_type).get('battery_decay_rate', None)
            battery_threshold = self.agent_config.get(agent_type).get('battery_threshold', None)

            for n in range(self.agent_config[agent_type]['num_agents']):
                self.agents.append(Agent3D(type = agent_type,
                                        agent_id = len(self.agents), 
                                        init_position = init_position[n],
                                        dt = self.dt, 
                                        vis_radius = self.vis_radius,
                                        map_resolution = map_resolution,
                                        config = self.agent_config[agent_type],
                                        controller_params = self.controller_params,
                                        path_planner = path_planner,
                                        path = paths[n] if paths!=[] else [],
                                        goal = goals[n] if goals is not None else None,
                                        init_battery = init_battery[n] if init_battery is not None else 0.5,
                                        battery_decay_rate = battery_decay_rate if battery_decay_rate is not None else None,
                                        battery_threshold = battery_threshold if battery_threshold is not None else 0.0,
                                        show_old_path = env.show_old_path))
                

    def compute_adjacency_matrix(self):
        """
        Computes the adjacency matrix for the swarm based on communication radius
        and line-of-sight checks in 3D space.
        
        Args:
            communication_radius (float): Maximum distance for communication.
            is_line_of_sight_free_fn (function): Function to check if the line of sight
                                                between two positions is obstacle-free.
        
        Returns:
            np.ndarray: Adjacency matrix indicating connections between agents.
        """
        # Use full 3D positions
        positions = np.array([agent.state[:3] for agent in self.agents])
        edge_obstacle = np.array([agent.obstacle_avoidance for agent in self.agents])
        num_agents = len(self.agents)
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):  # Check only upper triangular to avoid redundancy
                # Compute Euclidean distance in 3D
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= self.vis_radius:
                    if edge_obstacle[i] and edge_obstacle[j]:
                        # Perform line-of-sight check if obstacle avoidance is enabled
                        if self.is_line_of_sight_free_fn(positions[i], positions[j]):
                            adjacency_matrix[i, j] = 1
                            adjacency_matrix[j, i] = 1  # Symmetric adjacency matrix
                    else:
                        # Connect directly if obstacle avoidance is not required
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1  # Symmetric adjacency matrix

        return adjacency_matrix
