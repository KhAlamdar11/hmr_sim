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
        self.fiedler_value = None
        self.fiedler_vector = None
        self.critical_battery_level = self.params['critical_battery_level']


    def __call__(self, agent_idx, poses, A):
        """
        Compute the control input for the UAVs based on their positions and battery levels.

        Parameters:
        position (ndarray): Positions of the UAVs.
        batteries (ndarray): Battery levels of the UAVs.

        Returns:
        ndarray: Control input for the UAVs.
        """
        # neighbor_positions = []
        # for agent in agents:
        #     neighbor_positions.append(agent.get_pos())
        # print(neighbor_positions)
        # neighbor_positions = np.array(neighbor_positions)
        neighbor_positions = poses

        nv = neighbor_positions.shape[0]

        positionx = neighbor_positions[:, 0] 
        positiony = neighbor_positions[:, 1]

        self.fiedler_value = np.array(self.algebraic_connectivity(A))
        self.fiedler_vector = np.array(self.compute_eig_vector(A))

        dotx = np.zeros(nv)
        doty = np.zeros(nv)

        # Augment batteries to include pinned nodes
        # batteries = np.concatenate((np.array([0.7, 0.7]), batteries))

        for i in range(nv):
            for j in range(nv):
                if A[i, j] > 0:
                    if self.fiedler_value > self.params['epsilon']:
                        k = (-(1 / (self.params['sigma'] ** 2)) * (self.csch(self.fiedler_value - self.params['epsilon']) ** 2)) * (
                                A[i, j] * ((self.fiedler_vector[i] - self.fiedler_vector[j]) ** 2))
                    else:
                        k = -(1 / (self.params['sigma'] ** 2)) * 100 * (A[i, j] * ((self.fiedler_vector[i] - self.fiedler_vector[j]) ** 2))

                    # gain for battery aware controller
                    # batt_gain = self.battery_gain(batteries[j])

                    if self.params['battery_aware']:
                        dotx[i] += k * (positionx[i] - positionx[j]) #* batt_gain
                        doty[i] += k * (positiony[i] - positiony[j]) #* batt_gain
                    else:
                        dotx[i] += k * (positionx[i] - positionx[j])
                        doty[i] += k * (positiony[i] - positiony[j])

        u_c = np.column_stack([dotx, doty]) * self.params['gainConnectivity']  \
                                        + self.calculate_repulsion_forces(neighbor_positions)
        return u_c[agent_idx]


    def calculate_repulsion_forces(self, positions):
        """
        Calculate repulsion forces for positions within a threshold.

        Parameters:
        positions (ndarray): Positions of the UAVs.

        Returns:
        ndarray: Repulsion vectors for the UAVs.
        """
        threshold = self.params['repelThreshold']
        repulsion_strength = self.params['gainRepel']
        n = positions.shape[0]
        repulsion_vectors = np.zeros_like(positions)

        for i in range(n):
            for j in range(i + 1, n):
                difference_vector = positions[i] - positions[j]
                distance = np.linalg.norm(difference_vector)
                if distance < threshold:
                    repulsion_vector = difference_vector / distance
                    repulsion_vectors[i] += repulsion_vector * repulsion_strength
                    repulsion_vectors[j] -= repulsion_vector * repulsion_strength

        return repulsion_vectors


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