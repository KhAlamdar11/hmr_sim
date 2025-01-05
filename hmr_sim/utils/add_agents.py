from copy import deepcopy

import numpy as np

from hmr_sim.utils.utils import svstack, distance

"""
This class is responsible for adding new agents to a swarm based on predefined 
modes and criteria. The system parameters and modes are set during initialization, 
and a new agent is added by calling the class instance.

Attributes:
    mode (str): The mode of agent addition ('add_agent_base' or 'add_agent_near').
    type (str): Type of the agent to be added (for heterogeneous systems).
    agent_addition_id (int): ID of the agent used as a base for new agent placement.
    agents (list): List of existing agents in the swarm.
    comm_radius (float): Communication radius for agent interactions.
    neighbors (list): List of neighboring agents (for 'add_agent_near' mode).
"""


class AddAgent:
    def __init__(self, params, agents, comm_radius):

        # add_agent_base, add_agent_near
        self.mode = params['mode']

        # In heterogeneous systems, specify which kind of agent you are adding
        self.type = params['agent_type_to_add']

        # For Mode: add_agent_base -> ID of the agent where the new agent must be added
        self.agent_addition_id = params['agent_addition_id']

        self.agents = agents
        self.comm_radius = comm_radius

        self.seen_ids = set()

        # internal add agents params
        self.n_samples_per = 30

        # For add_agent_near only: Neighboring agents
        self.neighbors = None

        # Dispatch table for mode-to-method mapping
        self.mode_dispatch = {
            'add_agent_base': self.add_agent_base,
            'add_agent_near': self.add_agent_near,
        }

    def __call__(self):
        """ Adds agent to the network based on system parameters. """
        if self.mode in self.mode_dispatch:
            self.mode_dispatch[self.mode]()
        else:
            raise ValueError(f"Unknown add_agent mode: {self.mode}")

    def add_agent_base(self):

        # Find base position
        pos = self.find_position()
        print(f"base: {pos}")

        # Add new agent at position
        new_pos = self.add_agent_at(pos)
        print(f"Adding new agent at position: {new_pos}")

        self.create_new_agent(new_pos)

    def add_agent_near(self):

        arc = np.array([])
        poss = np.array([])

        for neigh in self.neighbors:
            neigh_pos = neigh.get_pos()
            res = self.add_agent_at(neigh_pos, return_all=True, factor_comm=0.37, factor_dist=0.3)
            if res is None:
                continue
            a, pos = res
            arc = svstack([arc, a])
            poss = np.concatenate([poss, pos])

        if poss.shape[0] != 0:
            m = np.argmax(poss)

            new_pos = np.array([arc[m, 0], arc[m, 1]])

            self.create_new_agent(new_pos)

    # ______________________  utils  _____________________

    def add_agent_at(self, pos, return_all=False, factor_comm=0.7, factor_dist=0.5):
        """
        Creates a new position for a new agent to be added in the vicinity of a given agent.

        Args:
            pos (numpy.ndarray): position of the agent in who's vicinity a new agent has to be added
            return_all (bool): flag to return all positions or only a selected one (different usage for add agent base and near)
            factor_comm (float): the factor of communication radius at which new points are to be sampled in the vicinity of pos
            factor_dist: distance factor of communication radius that serves as a min distance threshold for filtering out positions too close to other agents

        Returns:
            numpy.ndarray: position of the selected point if return_all = False
            OR
            numpy.ndarray: Position of all the possible sampled adn shortlisted points
            numpy.ndarray: Number of connection each point potentially makes with agents in the network
        """

        t = np.linspace(0, 2 * np.pi, self.n_samples_per)
        a, b = factor_comm * self.comm_radius * np.cos(t), factor_comm * self.comm_radius * np.sin(t)
        a += pos[0]
        b += pos[1]
        arc = np.array([[a, b] for a, b in zip(a, b)])

        # shortlist pts
        possible_pts = []
        possible_connections = []
        for j in range(arc.shape[0]):
            allow = True
            cons = 0
            for agent in self.agents:
                if distance(arc[j], agent.get_pos()) < self.comm_radius * factor_dist:
                    allow = False
                    break
                elif agent.agent_id != self.agent_addition_id and \
                        distance(arc[j], agent.get_pos()) < self.comm_radius:
                    if self.mode == 'add_agent_base':
                        cons += 1
                    elif self.mode == 'add_agent_near' and agent in self.neighbors:
                        cons += 1
            if allow:
                possible_pts.append(arc[j])
                possible_connections.append(cons)

        if len(possible_connections) != 0:
            arc, poss = np.array(possible_pts), np.array(possible_connections)
            m = np.argmax(poss)
        else:
            return None

        if return_all:
            return np.array(possible_pts), np.array(possible_connections)

        return np.array([arc[m, 0], arc[m, 1]])

    def create_new_agent(self, pos):
        """
        Creates a new agent of specified type of type and sets its parameters

        Args:
            pos: The new position where the agent must be added
        """
        agent = self.find_sample_agent()
        new_agent = deepcopy(agent)

        # find max if
        [self.seen_ids.add(agent.agent_id) for agent in self.agents]

        new_agent.agent_id = max(self.seen_ids) + 1

        new_agent.battery = 1.0
        new_agent.set_pos(pos)
        new_agent.old_path = []
        self.agents.append(new_agent)

    def find_sample_agent(self):
        """ Samples an agent of the type to be added so a new instance is not needed to be created from scratch """
        for agent in self.agents:
            if agent.type == self.type:
                return agent

    def find_position(self):
        """ Finds the position of the agent in whose vicinity a new agent has to be added. Used for the add agent base methodology to specify where the base is.       """
        for agent in self.agents:
            if agent.agent_id == self.agent_addition_id:
                return agent.get_pos()

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
