import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eig
from numpy.linalg import inv

"""
The original code structure is adapted from the work of Sabbatini, 2013.
The code is adapted from part of MATLAB codebase at https://github.com/Cinaraghedini/adaptive-topology
Battery awareness is included.
See code documentation and associated work for details.
"""

class ConnectivityController:

    def __init__(self, params):
        """
        Initialize the KConnectivityController with the given parameters.

        Parameters:
        params (dict): Configuration parameters for the controller.
        """
        self.params = params
        print(params)
        self.fiedler_value = None
        self.fiedler_vector = None
        self.critical_battery_level = self.params['critical_battery_level']


    def __call__(self, agent_idx, agent_position, neighbors, A, id_to_index):
        """
        Compute the control input for a specific UAV based on its position and neighbors.

        Parameters:
        agent_idx (int): Index of the agent for which to compute the control input.
        agent_position (ndarray): Position of the current UAV (1D array of shape (2,)).
        neighbor_positions (ndarray): Positions of the neighboring UAVs (2D array of shape (n_neighbors, 2)).

        Returns:
        ndarray: Control input for the agent.
        """

        print("A ", A)

        if A.size == 0:
            return np.array([0,0])

        # Compute Fiedler value and vector
        fiedler_value = self.algebraic_connectivity(A)  # Update if adjacency is required
        fiedler_vector = self.compute_eig_vector(A)  # Update if adjacency is required

        # Initialize control input
        control_vector = np.zeros(2)

        # Compute position differences and control contribution
        for agent in neighbors:

            neighbor_position = agent.get_pos()
            neighbor_id = id_to_index[agent.get_id()]

            dx = agent_position[0] - neighbor_position[0]
            dy = agent_position[1] - neighbor_position[1]

            # print(A.shape)
            # print(agent.agent_id)
            # print(fiedler_vector.shape)

            # Compute the interaction gain
            if fiedler_value > self.params['epsilon']:
                k = (-(1 / (self.params['sigma'] ** 2)) *
                    (self.csch(fiedler_value - self.params['epsilon']) ** 2)) * (
                        ((fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2))
            else:
                k = -(1 / (self.params['sigma'] ** 2)) * 100 * (((fiedler_vector[id_to_index[agent_idx]] - fiedler_vector[neighbor_id]) ** 2))

            # Accumulate control contributions
            if agent.type == 2:
                control_vector[0] += k * dx * 3 
                control_vector[1] += k * dy * 3
            else:
                control_vector[0] += k * dx 
                control_vector[1] += k * dy


        # Scale control input
        control_vector = control_vector * self.params['gainConnectivity'] \
                                        + self.calculate_repulsion_forces(agent_position, neighbors)

        # return control_vector

        return np.clip(control_vector, -0.3, 0.3)


    def calculate_repulsion_forces(self, agent_position, neighbor_positions):
        """
        Calculate repulsion forces for a specific agent based on its neighbors.

        Parameters:
        agent_position (ndarray): Position of the current agent (1D array of shape (2,)).
        neighbor_positions (ndarray): Positions of the neighboring UAVs (2D array of shape (n_neighbors, 2)).

        Returns:
        ndarray: Repulsion vector for the agent.
        """
        threshold = self.params['repelThreshold']
        repulsion_strength = self.params['gainRepel']
        repulsion_vector = np.zeros(2)

        for agent in neighbor_positions:
            neighbor_position = agent.get_pos()
            difference_vector = agent_position - neighbor_position
            distance = np.linalg.norm(difference_vector)
            if 0 < distance < threshold:  # Avoid division by zero
                unit_vector = difference_vector / distance
                repulsion_vector += unit_vector * repulsion_strength

        return repulsion_vector



    def csch(self, x):
        """Compute the hyperbolic cosecant of x."""
        return 1.0 / np.sinh(x)


    def degree(self, A):
        """Compute the degree matrix of adjacency matrix A."""
        return np.diag(np.sum(A, axis=1))


    def algebraic_connectivity(self, A):
        """
        Calculate the algebraic connectivity of the adjacency matrix A.

        Parameters:
        A (ndarray): Adjacency matrix.

        Returns:
        float: Algebraic connectivity value.
        """
        D = self.degree(A)
        if np.all(np.diag(D) != 0):
            L = D - A
            if self.params['normalized']:
                D_inv_sqrt = inv(np.sqrt(D))
                L = D_inv_sqrt @ L @ D_inv_sqrt
            eValues, _ = eig(L)
            eValues = np.sort(eValues.real)
            ac = eValues[1]
        else:
            ac = 0
        return ac


    def compute_eig_vector(self, A):
        """
        Compute the eigenvector corresponding to the second smallest eigenvalue of the Laplacian matrix.

        Parameters:
        A (ndarray): Adjacency matrix.

        Returns:
        ndarray: Eigenvector corresponding to the second smallest eigenvalue.
        """
        D = self.degree(A)
        L = D - A
        if self.params['normalized']:
            D_inv_sqrt = inv(np.sqrt(D))
            L = D_inv_sqrt @ L @ D_inv_sqrt
        eValues, eVectors = eig(L)
        Y = np.argsort(eValues.real)
        v = eVectors[:, Y[1]]
        return v.real
    

    def clip(self, velocities):
        """
        Clip velocities to ensure they do not exceed the maximum allowed velocity.
        """
        magnitudes = np.linalg.norm(velocities, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factors = np.where(magnitudes > self.params['v_max'], self.params['v_max'] / magnitudes, 1)
        scaled_velocities = velocities * scale_factors[:, np.newaxis]
        return scaled_velocities


    def battery_gain(self, b):
        """
        Calculate the battery gain based on the current battery level.

        Parameters:
        b (float): Current battery level.

        Returns:
        float: Battery gain.
        """
        return np.exp((self.critical_battery_level - b) / self.params['tau']) + 1