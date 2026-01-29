import numpy as np
from copy import deepcopy

"""
CentralizedExplorationController coordinates multi-agent frontier exploration.

This controller assigns frontier goals to exploring agents using either a greedy
nearest-first strategy or an optimal Hungarian assignment strategy. It prevents
multiple agents from being assigned to the same frontier.
"""


class CentralizedExplorationController:
    def __init__(self, frontier_detector, assignment_strategy='greedy_nearest'):
        """
        Initialize the centralized exploration controller.

        Args:
            frontier_detector: FrontierDetector instance for detecting frontiers.
            assignment_strategy (str): Assignment strategy - 'greedy_nearest' or 'hungarian'.
        """
        self.frontier_detector = frontier_detector
        self.assignment_strategy = assignment_strategy
        self.agent_assignments = {}  # agent_id -> goal position
        self.assigned_frontiers = set()  # Set of assigned frontier tuple positions
        self.all_frontiers = []  # List of all current frontier positions

    def update(self, exploration_map, agents):
        """
        Update frontier assignments for all agents.

        Args:
            exploration_map (np.ndarray): Current exploration map.
            agents (list): List of Agent objects using centralized_explore controller.

        Returns:
            dict: Mapping of agent_id -> goal position.
        """
        # Detect frontiers from exploration map
        self.frontier_detector.set_map(exploration_map)
        frontier_map = self.frontier_detector.detect_frontiers()
        candidate_points, _ = self.frontier_detector.label_frontiers(frontier_map)

        # Convert candidate points to world coordinates
        frontiers = []
        for point in candidate_points:
            # Access the dunder method directly (it has __ on both sides, not just leading)
            world_pos = self.frontier_detector.__map_to_position__(point)
            frontiers.append(np.array(world_pos))

        self.all_frontiers = frontiers

        # Find agents needing reassignment
        agents_needing_assignment = []
        for agent in agents:
            if agent.controller_type != 'centralized_explore':
                continue

            agent_id = agent.get_id()
            current_goal = self.agent_assignments.get(agent_id)

            needs_assignment = False

            # No current goal
            if current_goal is None:
                needs_assignment = True
            else:
                # Check if agent reached its goal
                distance_to_goal = np.linalg.norm(agent.get_pos() - current_goal)
                if distance_to_goal < 0.5:  # Goal reached threshold
                    needs_assignment = True
                    # Remove from assigned frontiers
                    goal_tuple = tuple(current_goal)
                    if goal_tuple in self.assigned_frontiers:
                        self.assigned_frontiers.discard(goal_tuple)

                # Check if assigned frontier is still valid
                if not needs_assignment and len(frontiers) > 0:
                    min_dist_to_frontier = min(
                        np.linalg.norm(current_goal - f) for f in frontiers
                    )
                    # If the assigned frontier no longer exists (explored)
                    if min_dist_to_frontier > 1.0:
                        needs_assignment = True
                        goal_tuple = tuple(current_goal)
                        if goal_tuple in self.assigned_frontiers:
                            self.assigned_frontiers.discard(goal_tuple)

            if needs_assignment:
                agents_needing_assignment.append(agent)
                # Clear old assignment
                if agent_id in self.agent_assignments:
                    del self.agent_assignments[agent_id]

        # Get available frontiers (not already assigned)
        available_frontiers = []
        for f in frontiers:
            f_tuple = tuple(f)
            is_assigned = False
            for assigned_f in self.assigned_frontiers:
                if np.linalg.norm(np.array(assigned_f) - f) < 0.5:
                    is_assigned = True
                    break
            if not is_assigned:
                available_frontiers.append(f)

        # Assign frontiers using selected strategy
        if len(agents_needing_assignment) > 0 and len(available_frontiers) > 0:
            if self.assignment_strategy == 'greedy_nearest':
                new_assignments = self._greedy_nearest_assignment(
                    agents_needing_assignment, available_frontiers
                )
            elif self.assignment_strategy == 'hungarian':
                new_assignments = self._hungarian_assignment(
                    agents_needing_assignment, available_frontiers
                )
            else:
                new_assignments = self._greedy_nearest_assignment(
                    agents_needing_assignment, available_frontiers
                )

            # Update assignments
            for agent_id, goal in new_assignments.items():
                self.agent_assignments[agent_id] = goal
                self.assigned_frontiers.add(tuple(goal))

        return deepcopy(self.agent_assignments)

    def _greedy_nearest_assignment(self, agents, frontiers):
        """
        Greedy nearest-first assignment strategy.

        Assigns each agent to its nearest available frontier, one at a time.

        Args:
            agents (list): List of Agent objects needing assignment.
            frontiers (list): List of available frontier positions.

        Returns:
            dict: Mapping of agent_id -> goal position.
        """
        assignments = {}
        available = list(frontiers)

        for agent in agents:
            if len(available) == 0:
                break

            agent_pos = agent.get_pos()
            agent_id = agent.get_id()

            # Find nearest frontier
            min_dist = float('inf')
            nearest_idx = -1
            for i, frontier in enumerate(available):
                dist = np.linalg.norm(agent_pos - frontier)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

            if nearest_idx >= 0:
                assignments[agent_id] = available[nearest_idx]
                available.pop(nearest_idx)

        return assignments

    def _hungarian_assignment(self, agents, frontiers):
        """
        Optimal Hungarian algorithm assignment strategy.

        Uses the Hungarian algorithm (linear_sum_assignment) to find the
        optimal assignment that minimizes total travel distance.

        Args:
            agents (list): List of Agent objects needing assignment.
            frontiers (list): List of available frontier positions.

        Returns:
            dict: Mapping of agent_id -> goal position.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            print("Warning: scipy not available, falling back to greedy assignment")
            return self._greedy_nearest_assignment(agents, frontiers)

        if len(agents) == 0 or len(frontiers) == 0:
            return {}

        # Build cost matrix (distances)
        n_agents = len(agents)
        n_frontiers = len(frontiers)
        cost_matrix = np.zeros((n_agents, n_frontiers))

        for i, agent in enumerate(agents):
            agent_pos = agent.get_pos()
            for j, frontier in enumerate(frontiers):
                cost_matrix[i, j] = np.linalg.norm(agent_pos - frontier)

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = {}
        for i, j in zip(row_ind, col_ind):
            if j < len(frontiers):  # Ensure valid frontier index
                agent_id = agents[i].get_id()
                assignments[agent_id] = frontiers[j]

        return assignments

    def get_agent_goal(self, agent_id):
        """
        Get the assigned goal for a specific agent.

        Args:
            agent_id (int): The agent's ID.

        Returns:
            np.ndarray or None: The assigned goal position or None.
        """
        return self.agent_assignments.get(agent_id)

    def get_all_assignments(self):
        """
        Get all current agent-goal assignments.

        Returns:
            dict: Copy of the agent_id -> goal mapping.
        """
        return deepcopy(self.agent_assignments)

    def get_all_frontiers(self):
        """
        Get all detected frontiers from the last update.

        Returns:
            list: List of frontier positions.
        """
        return self.all_frontiers

    def clear_assignment(self, agent_id):
        """
        Clear the assignment for a specific agent.

        Args:
            agent_id (int): The agent's ID.
        """
        if agent_id in self.agent_assignments:
            goal = self.agent_assignments[agent_id]
            goal_tuple = tuple(goal)
            if goal_tuple in self.assigned_frontiers:
                self.assigned_frontiers.discard(goal_tuple)
            del self.agent_assignments[agent_id]
