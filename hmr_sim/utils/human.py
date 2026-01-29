import numpy as np

"""
Represents a human that can be detected by agents in the simulation.

This class models a stationary human at a predefined position that can be
detected by agents with a directional FOV (field of view).

Attributes:
    human_id (int): Unique identifier for the human.
    position (np.ndarray): The [x, y] position of the human.
    detected (bool): Whether the human has been detected.
    detected_by (int or None): ID of the agent that detected this human.
    detection_time (float or None): Simulation time when detected.
"""


class Human:
    def __init__(self, human_id, position):
        """
        Initialize a Human instance.

        Args:
            human_id (int): Unique identifier for the human.
            position (list or np.ndarray): The [x, y] position of the human.
        """
        self.human_id = human_id
        self.position = np.array(position, dtype=np.float64)
        self.detected = False
        self.detected_by = None
        self.detection_time = None

    def get_position(self):
        """
        Returns the position of the human.

        Returns:
            np.ndarray: The [x, y] position.
        """
        return self.position

    def mark_detected(self, agent_id, time):
        """
        Mark this human as detected by an agent.

        Args:
            agent_id (int): ID of the detecting agent.
            time (float): Simulation time of detection.
        """
        if not self.detected:
            self.detected = True
            self.detected_by = agent_id
            self.detection_time = time
            print(f"Human {self.human_id} detected by Agent {agent_id} at time {time}")

    def is_detected(self):
        """
        Check if this human has been detected.

        Returns:
            bool: True if detected, False otherwise.
        """
        return self.detected

    def get_id(self):
        """
        Returns the human's unique identifier.

        Returns:
            int: The human ID.
        """
        return self.human_id

    def __repr__(self):
        status = "detected" if self.detected else "undetected"
        return f"Human(id={self.human_id}, pos={self.position}, {status})"
