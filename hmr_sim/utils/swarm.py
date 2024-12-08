from hmr_sim.utils.agent import Agent
import numpy as np
from scipy.spatial.distance import euclidean
import math
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.rrt import RRT


class Swarm:
    """Manages a swarm of heterogeneous agents."""

    def __init__(self, env, config, map_resolution, map_handlers):
        
        self.action_dim = 2  # Assuming 2D action space

        self.vis_radius = config.get('vis_radius')

        self.dt = config.get('dt')

        self.agent_config = config.get('agent_config')

        #________________________  controller  ________________________
        controller_params = config.get('controller_params')
        sigma = math.sqrt(-self.vis_radius/(2*math.log(controller_params['delta'])))
        controller_params['sigma'] = sigma
        controller_params['range'] = self.vis_radius
        controller_params['repelThreshold'] = self.vis_radius*controller_params['repelThreshold']

        self.total_agents = self.total_agents = sum(inner_dict["num_agents"] for inner_dict in self.agent_config.values())

        #________________________  map handlers  ________________________
        self.is_line_of_sight_free_fn = map_handlers['is_line_of_sight_free']
        self.is_free_space_fn = map_handlers['is_free_space']
        self.update_exploration_map_fn = map_handlers['update_exploration_map']
        self.get_frontier_goal_fn = map_handlers['get_frontier_goal']

        #________________________  instance definitions  ________________________
        self.agents = []
        id_n = 0
        for agent_type in self.agent_config.keys():
            # positions need to be handled here for the group as a whole
            init_position = self.agent_config[agent_type]['init_position']
            # print(f"Agent Type: {agent_type}")
            # print(f"pos: {init_position}, form: {init_formation}")
            if init_position == 'None':
                init_formation = self.agent_config[agent_type]['init_formation']
                # If init position is empty, create the formation
                print(f"Using initialization formation: {init_formation}")
                init_position = get_curve(init_formation, 
                                          self.agent_config[agent_type]['num_agents'])   
                
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
                        
                # baterries
                init_battery = self.agent_config.get(agent_type).get('init_battery', None)
                battery_decay_rate = self.agent_config.get(agent_type).get('battery_decay_rate', None)

            for n in range(self.agent_config[agent_type]['num_agents']):
                self.agents.append(Agent(type = agent_type,
                                        agent_id = len(self.agents), 
                                        init_position = init_position[n],
                                        dt = self.dt, 
                                        vis_radius = self.vis_radius,
                                        map_resolution = map_resolution,
                                        config = self.agent_config[agent_type],
                                        controller_params = controller_params,
                                        path_planner = path_planner,
                                        path = paths[n] if paths!=[] else [],
                                        goal = goals[n] if goals is not None else None,
                                        init_battery = init_battery[n] if init_battery is not None else None,
                                        battery_decay_rate = battery_decay_rate if battery_decay_rate is not None else None))
               

    def get_states(self):
        return np.array([agent.state for agent in self.agents])


    def get_poses(self):
        return np.array([agent.state[:2] for agent in self.agents])


    def step(self, actions, is_free_space_fn):
        pass
        # for agent, action in zip(self.agents, actions):
        #     agent.update_state(self, action, is_free_space_fn)


    def compute_adjacency_matrix(self):
        """
        Computes the adjacency matrix for the swarm based on communication radius
        and line-of-sight checks.
        
        Args:
            communication_radius (float): Maximum distance for communication.
            is_line_of_sight_free_fn (function): Function to check if the line of sight
                                                 between two positions is obstacle-free.
        
        Returns:
            np.ndarray: Adjacency matrix indicating connections between agents.
        """
        positions = np.array([agent.state[:2] for agent in self.agents])
        edge_osbtacle = np.array([agent.obstacle_avoidance for agent in self.agents])
        num_agents = len(self.agents)
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):  # Check only upper triangular to avoid redundancy
                distance = euclidean(positions[i], positions[j])
                if distance <= self.vis_radius:
                    if edge_osbtacle[i] and edge_osbtacle[j]:
                        if self.is_line_of_sight_free_fn(positions[i], positions[j]):
                            adjacency_matrix[i, j] = 1
                            adjacency_matrix[j, i] = 1  # Symmetric adjacency matrix
                    else:
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1  # Symmetric adjacency matrix

        return adjacency_matrix


    def get_dummy_action(self):
        num_agents = len(self.agents)
        action_dim = 2  # Assuming a 2D action space
        return np.zeros((num_agents, action_dim), dtype=float)


    def update_neighbors(self):
        """
        Computes neighbors for each agent based on the adjacency matrix and updates them.
        """
        adjacency_matrix = self.compute_adjacency_matrix()
        for i, agent in enumerate(self.agents):
            neighbors = [self.agents[j] for j in range(len(self.agents)) if adjacency_matrix[i, j] == 1]
            agent.set_neighbors(neighbors)


    def set_agent_path(self,idx,path):
        self.agents[idx].set_path(path)


    def get_paths(self):
        return self.paths
    

    def run_controllers(self):
        self.update_neighbors()
        for agent in self.agents:
            agent.run_controller(self)
            # Update battery
            # if battery < battery_threshold:
            #     self.remove_agent()