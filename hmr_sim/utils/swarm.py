import math

import numpy as np
from scipy.linalg import eig
from scipy.spatial.distance import euclidean

from hmr_sim.utils.add_agents import AddAgent
from hmr_sim.utils.agent import Agent
from hmr_sim.utils.lattice_generation import gen_lattice
from hmr_sim.utils.rrt import RRT
from hmr_sim.utils.utils import get_curve

"""
The Swarm class manages a collection of heterogeneous agents in a multi-agent system.
This class handles the initialization, simulation, and control of agents in a swarm, 
including path planning, battery management, neighbor updates, and connectivity checks. 
It supports dynamic addition and removal of agents based on battery levels or user-defined 
criteria.
"""


class Swarm:
    def __init__(self, env, config, map_resolution, map_handlers, is_initialize_swarm=True):

        self.env = env
        self.action_dim = 2  # Assuming 2D action space
        self.vis_radius = config.get('vis_radius')
        self.dt = config.get('dt')

        self.agent_config = config.get('agent_config')

        # ________________________  controller  ________________________
        self.controller_params = config.get('controller_params')
        self.controller_params['sigma'] = math.sqrt(-self.vis_radius / (2 * math.log(self.controller_params['delta'])))
        self.controller_params['range'] = self.vis_radius
        self.controller_params['repelThreshold'] = self.vis_radius * self.controller_params['repelThreshold']

        self.total_agents = self.total_agents = sum(
            inner_dict["num_agents"] for inner_dict in self.agent_config.values())

        # ________________________  map handlers  ________________________
        self.is_line_of_sight_free_fn = map_handlers['is_line_of_sight_free']
        self.is_free_space_fn = map_handlers['is_free_space']
        self.update_exploration_map_fn = map_handlers['update_exploration_map']
        self.get_frontier_goal_fn = map_handlers['get_frontier_goal']

        self.agents = []

        # ________________________  instance definitions  ________________________
        if is_initialize_swarm:
            goals = None
            path_planner = None
            paths = []
            for agent_type in self.agent_config.keys():

                # __________________  Initial position handling  ___________________
                init_position = self.agent_config[agent_type]['init_position']
                if init_position == 'None':
                    init_formation = self.agent_config[agent_type]['init_formation']
                    print(f"Using initialization formation: {init_formation}")
                    if init_formation['shape'] != 'lattice':
                        init_position = get_curve(init_formation,
                                                  self.agent_config[agent_type]['num_agents'])
                    else:
                        start = np.array([0.0, 0.0])
                        end = np.array([3.5, 0.0])
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
                # Battery handling
                init_battery = self.agent_config.get(agent_type).get('init_battery', None)
                if init_battery == 'autofill':
                    init_battery = np.linspace(0.16, 0.9, self.agent_config[agent_type]['num_agents'])
                battery_decay_rate = self.agent_config.get(agent_type).get('battery_decay_rate', None)
                battery_threshold = self.agent_config.get(agent_type).get('battery_threshold', None)

                for n in range(self.agent_config[agent_type]['num_agents']):
                    self.agents.append(Agent(type=agent_type,
                                             agent_id=len(self.agents),
                                             init_position=init_position[n],
                                             dt=self.dt,
                                             vis_radius=self.vis_radius,
                                             map_resolution=map_resolution,
                                             config=self.agent_config[agent_type],
                                             controller_params=self.controller_params,
                                             path_planner=path_planner,
                                             path=paths[n] if paths != [] else [],
                                             goal=goals[n] if goals is not None else None,
                                             init_battery=init_battery[n] if init_battery is not None else 0.5,
                                             battery_decay_rate=battery_decay_rate if battery_decay_rate is not None else None,
                                             battery_threshold=battery_threshold if battery_threshold is not None else 0.0,
                                             show_old_path=env.show_old_path))

        # ________________________  Agent Additions  ________________________
        if self.agents:
            self.add_agent_params = config.get('add_agent_params')
            self.add_agent = AddAgent(self.add_agent_params, self.agents, self.vis_radius)
            self.add_agent_already_added = []

        self.fiedler_list = []
        self.n_agents_list = []

    def compute_adjacency_matrix(self):
        positions = np.array([agent.state[:2] for agent in self.agents])
        edge_osbtacle = np.array([agent.is_obstacle_avoidance for agent in self.agents])
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

    def run_controllers(self):
        """
        Executes the controllers for all agents and manages agent additions/removals.
        This method updates neighbors, runs controllers for active agents, and dynamically
        adds or removes agents based on battery thresholds or user-defined criteria.
        """

        self.update_neighbors()
        # self.save_fiedler_value()

        to_remove = []
        to_add = []

        all_active = self.all_active()

        for agent in self.agents:  # Create a shallow copy for iteration
            if all_active or agent.type == 'UAV':
                agent.run_controller(self)

        for agent in self.agents:  # Create a shallow copy for iteration            
            # Check battery is below first thresold and then add a new agent
            if agent.battery < self.add_agent_params['battery_of_concern'] and \
                    agent not in self.add_agent_already_added:
                to_add.append(agent)
                self.add_agent_already_added.append(agent)

            # Check battery and remove agent if it falls below the threshold
            if agent.is_battery_critical():
                to_remove.append(agent)

        for agent in to_add:
            self.add_agent_swarm(agent, self.add_agent_already_added)

        for agent in to_remove:
            self.add_agent_already_added.remove(agent)
            self.remove_agent(agent)  # Call the method to remove the agent

    def all_active(self):
        for agent in self.agents:
            if agent.mode != 'active':
                return False
        return True

    def remove_agent(self, agent):
        self.agents.remove(agent)
        self.total_agents -= 1
        print(f"Agent {agent.agent_id} removed due to low battery.")

    def add_agent_swarm(self, agent, to_remove):
        """
        Dynamically adds an agent to the swarm based on the specified criteria.

        Args:
            agent (Agent): The agent triggering the addition.
            to_remove (list): List of agents to be removed (optional).
        """

        self.add_agent.set_neighbors(agent.neighbors)

        # if add_agent_criteria == 'min_fiedler':
        #     adjacency_matrix = self.compute_adjacency_matrix()
        #     if self.compute_fiedler_value(A) <= self.min_fiedler:
        #         add_agent(add_agent_mode, add_agent_criterion, self.agents)

        if self.add_agent_params['criterion'] == 'min_n_agents':
            if self.total_agents - len(to_remove) <= self.add_agent_params['critical_value']:
                print(f"Number of agents is {self.total_agents}, which is <= {self.add_agent_params['critical_value']}")
                self.add_agent()
                self.total_agents += 1

    def get_dummy_action(self):
        num_agents = len(self.agents)
        action_dim = 2  # Assuming a 2D action space
        return np.zeros((num_agents, action_dim), dtype=float)

    def update_neighbors(self):
        """ Updates the neighbors for each agent in the swarm based on the adjacency matrix. """
        adjacency_matrix = self.compute_adjacency_matrix()
        for i, agent in enumerate(self.agents):
            neighbors = [self.agents[j] for j in range(len(self.agents)) if adjacency_matrix[i, j] == 1]
            agent.set_neighbors(neighbors)

    def set_agent_path(self, idx, path):
        self.agents[idx].set_path(path)

    def get_paths(self):
        return self.paths

    def degree(self, A):
        """Compute the degree matrix of adjacency matrix A."""
        return np.diag(np.sum(A, axis=1))

    def get_states(self):
        return np.array([agent.state for agent in self.agents])

    def get_poses(self):
        return np.array([agent.state[:2] for agent in self.agents])

    def step(self, actions, is_free_space_fn):
        pass
        # for agent, action in zip(self.agents, actions):
        #     agent.update_state(self, action, is_free_space_fn)

    # ______________________  TESTS  ___________________

    def show_neighbors_per_agent(self):
        print('======================================')
        for agent in self.agents:
            print(f'Neighbors of agent {agent.agent_id} of type {agent.type} at {agent.get_pos()}')
            for neigh in agent.neighbors:
                print(f'Neighbors ID {neigh.agent_id} of type {neigh.type} at {neigh.get_pos()}')
            print('-----------------------------------')

    def save_fiedler_value(self):
        A = self.compute_adjacency_matrix()
        D = self.degree(A)

        if np.all(np.diag(D) != 0):
            L = D - A
            eValues, _ = eig(L)
            eValues = np.sort(eValues.real)
            ac = eValues[1]
        else:
            ac = 0

        self.fiedler_list.append(ac)
        self.n_agents_list.append(self.total_agents)

        # Save the two lists as .npy files
        np.save("fiedler_list.npy", np.array(self.fiedler_list))
        np.save("n_agents_list.npy", np.array(self.n_agents_list))
        np.save(f"/As/A_{int(self.total_agents)}.npy", A)
