from hmr_sim.utils.agent import Agent
import numpy as np
from scipy.spatial.distance import euclidean
import math
from hmr_sim.utils.utils import get_curve
from hmr_sim.utils.rrt import RRT
from hmr_sim.utils.add_agents import AddAgent
from hmr_sim.utils.lattice_generation import gen_lattice

class Swarm:
    """Manages a swarm of heterogeneous agents."""

    def __init__(self, env, config, map_resolution, map_handlers):
        
        self.env = env

        self.action_dim = 2  # Assuming 2D action space

        self.vis_radius = config.get('vis_radius')

        self.dt = config.get('dt')

        self.agent_config = config.get('agent_config')

        #________________________  controller  ________________________
        self.controller_params = config.get('controller_params')
        sigma = math.sqrt(-self.vis_radius/(2*math.log(self.controller_params['delta'])))
        self.controller_params['sigma'] = sigma
        self.controller_params['range'] = self.vis_radius
        self.controller_params['repelThreshold'] = self.vis_radius*self.controller_params['repelThreshold']

        self.total_agents = self.total_agents = sum(inner_dict["num_agents"] for inner_dict in self.agent_config.values())

        #________________________  map handlers  ________________________
        self.is_line_of_sight_free_fn = map_handlers['is_line_of_sight_free']
        self.is_free_space_fn = map_handlers['is_free_space']
        self.update_exploration_map_fn = map_handlers['update_exploration_map']
        self.get_frontier_goal_fn = map_handlers['get_frontier_goal']


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
            # baterries
            init_battery = self.agent_config.get(agent_type).get('init_battery', None)
            if init_battery=='autofill':
                init_battery = np.linspace(0.2, 0.9, self.agent_config[agent_type]['num_agents'])        
            battery_decay_rate = self.agent_config.get(agent_type).get('battery_decay_rate', None)
            battery_threshold = self.agent_config.get(agent_type).get('battery_threshold', None)

            for n in range(self.agent_config[agent_type]['num_agents']):
                self.agents.append(Agent(type = agent_type,
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


        #________________________  Agent Additions  ________________________
        # self.add_agent_criteria = 'min_n_agents'
        # self.min_n_agents = 14 + 2
        # self.add_agent_mode = 'add_agent_base'
        self.add_agent_params = config.get('add_agent_params')
        self.add_agent = AddAgent(self.add_agent_params, self.agents, self.vis_radius)

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

        to_remove = []
        to_add = []

        for agent in self.agents:  # Create a shallow copy for iteration
            agent.run_controller(self)
            
            # Check battery is below first thresold and then add a new agent
            if agent.battery < self.add_agent_params['battery_of_concern']:
                to_add.append(agent)

            # Check battery and remove agent if it falls below the threshold
            if agent.is_battery_critical(): 
                to_remove.append(agent)

        for agent in to_add:
            self.add_agent_swarm(agent)

        for agent in to_remove:
            self.remove_agent(agent)  # Call the method to remove the agent

    def remove_agent(self, agent):
        self.agents.remove(agent)
        self.total_agents -= 1
        print(f"Agent {agent.agent_id} removed due to low battery.")


    def add_agent_swarm(self,agent):

        self.add_agent.set_neighbors(agent.neighbors)

        # if add_agent_criteria == 'min_fiedler':
        #     adjacency_matrix = self.compute_adjacency_matrix()
        #     if self.compute_fiedler_value(A) <= self.min_fiedler:
        #         add_agent(add_agent_mode, add_agent_criterion, self.agents)
        
        # el
        if self.add_agent_params['criterion'] == 'min_n_agents':
            if self.total_agents <= self.add_agent_params['critical_value']:
                print(f"Number of agents is {self.total_agents}, which is <= {self.add_agent_params['critical_value']}")
                self.add_agent()
                self.total_agents += 1

    #______________________  TESTS  ___________________

    def show_neighbors_per_agent(self):
        print('======================================')
        for agent in self.agents:
            print(f'Neighbors of agent {agent.agent_id} of type {agent.type} at {agent.get_pos()}')
            for neigh in agent.neighbors:
                print(f'Neighbors ID {neigh.agent_id} of type {neigh.type} at {neigh.get_pos()}')
            print('-----------------------------------')
            