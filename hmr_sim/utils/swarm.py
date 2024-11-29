from hmr_sim.utils.agent import Agent
import numpy as np
from scipy.spatial.distance import euclidean
import math
from hmr_sim.utils.utils import get_curve

class Swarm:
    '''Manages a swarm of agents.'''

    def __init__(self, num_agents, init_positions, speed, dt, vis_radius, is_line_of_sight_free_fn):
        
        self.agents = [
            Agent(agent_id=i, init_pos=init_positions[i], speed=speed, dt=dt, vis_radius=vis_radius)
            for i in range(num_agents)
        ]
        
        self.action_dim = 2 

        self.vis_radius = vis_radius
        self.is_line_of_sight_free_fn = is_line_of_sight_free_fn
        

    def get_states(self):
        return np.array([agent.state for agent in self.agents])

    def get_poses(self):
        return np.array([agent.state[:2] for agent in self.agents])

    def step(self, actions, is_free_space_fn):
        for agent, action in zip(self.agents, actions):
            agent.update_state(self, action, is_free_space_fn)


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
        num_agents = len(self.agents)
        adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)

        for i in range(num_agents):
            for j in range(i + 1, num_agents):  # Check only upper triangular to avoid redundancy
                distance = euclidean(positions[i], positions[j])
                if distance <= self.vis_radius and self.is_line_of_sight_free_fn(positions[i], positions[j]):
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
            # print("-------------------")
            # print(agent.agent_id)
            # print(agent.get_neighbors_pos())
            # print("-------------------")

    def set_agent_path(self,idx,path):
        self.agents[idx].set_path(path)

    def get_paths(self):
        return self.paths
    
    def run_controllers(self):
        self.update_neighbors()
        for agent in self.agents:
            agent.run_controller(self)


class HeterogeneousSwarm(Swarm):
    """Manages a swarm of heterogeneous agents."""

    def __init__(self, num_agents, init_positions, speed, dt, vis_radius, agent_types, path_planners, is_line_of_sight_free_fn, config):
        
        self.action_dim = 2  # Assuming 2D action space
        self.is_line_of_sight_free_fn = is_line_of_sight_free_fn

        self.vis_radius = vis_radius

        self.speed = speed

        self.dt = dt

        #________________________  controller  ________________________
        controller_params = config.get('controller_params')
        print("delta ",controller_params['delta'])
        sigma = math.sqrt(-self.vis_radius/(2*math.log(controller_params['delta'])))
        controller_params['sigma'] = sigma
        controller_params['range'] = self.vis_radius
        controller_params['repelThreshold'] = self.vis_radius*controller_params['repelThreshold']

        self.num_agents = num_agents
        self.total_agents = np.sum(self.num_agents)

        self.types = np.repeat(np.arange(len(num_agents)),num_agents)

        # print(speed[self.types[0]])

        self.agents = [
            Agent(agent_id = i, 
                  init_pos = init_positions[i], 
                  speed = speed[self.types[i]], 
                  dt = dt, 
                  vis_radius = vis_radius,
                  type = self.types[i],
                  controller_type = agent_types[self.types[i]],
                  path_planner = path_planners[i],
                  controller_params = controller_params)
            for i in range(self.total_agents)
        ]

    def set_all_paths(self,paths):
        for idx in paths.keys():
            if self.agents[idx].controller_type == 'path_tracker':
                path = get_curve(paths[idx], speed=self.speed[self.agents[idx].type], dt=self.dt)
                self.agents[idx].set_path(path)