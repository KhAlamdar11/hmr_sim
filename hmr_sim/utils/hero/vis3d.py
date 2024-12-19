from vispy import scene, app
import numpy as np
import time

class SwarmRenderer3D:
    def __init__(self, swarm):
        """
        Initialize the 3D renderer for the swarm.

        Args:
            swarm: An object containing agents with positions and adjacency matrix.
        """
        self.swarm = swarm

        # Create a VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
        self.view = self.canvas.central_widget.add_view()

        # Configure the camera
        self.view.camera = scene.cameras.TurntableCamera(
            fov=60, azimuth=30, elevation=30, distance=15, center=(0, 0, 0)
        )

        # Store visuals
        self.markers = []  # List to store marker visuals
        self.adjacency_lines = []  # List to store adjacency lines

    def add_marker(self, position, radius=0.3, color=(0.2, 0.6, 1.0, 1.0)):
        """
        Add a spherical marker to the scene.

        Args:
            position: A tuple (x, y, z) specifying the marker's position.
            radius: Radius of the sphere.
            color: RGBA color of the sphere.
        """
        sphere = scene.visuals.Sphere(
            radius=radius,
            method='latitude',
            color=color,
            shading='smooth',  # Add smooth shading
            parent=self.view.scene
        )
        sphere.transform = scene.transforms.MatrixTransform()
        sphere.transform.translate(position)
        self.markers.append(sphere)

    def update_markers(self):
        """Efficiently update markers."""
        positions = np.array([agent.state[:3] for agent in self.swarm.agents])

        if len(self.markers) != len(positions):
            # If the number of agents changes, recreate markers
            for marker in self.markers:
                marker.parent = None
            self.markers.clear()

            for position in positions:
                self.add_marker(position)
        else:
            # Update positions of existing markers
            for i, marker in enumerate(self.markers):
                marker.transform.reset()  # Reset transform
                marker.transform.translate(positions[i])  # Update position


    def update_adjacency_lines(self):
        """Update or draw adjacency lines in a single batch."""
        adjacency_matrix = self.swarm.compute_adjacency_matrix()
        positions = np.array([agent.state[:3] for agent in self.swarm.agents])

        # Compute all line positions
        line_positions = []
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i + 1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j]:
                    line_positions.append([positions[i], positions[j]])

        # Flatten the line positions into a single array
        if len(line_positions) > 0:
            line_positions = np.array(line_positions).reshape(-1, 3)
            if hasattr(self, 'batched_lines'):
                self.batched_lines.set_data(pos=line_positions, color='yellow', width=2)
            else:
                self.batched_lines = scene.visuals.Line(
                    pos=line_positions,
                    color='yellow',
                    width=2,
                    parent=self.view.scene,
                )
        elif hasattr(self, 'batched_lines'):
            self.batched_lines.parent = None  # Remove from scene
            del self.batched_lines






    def update_scene(self):
        start_time = time.time()
        self.update_markers()
        # print(f"Update Markers Time: {time.time() - start_time:.6f}s")

        start_time = time.time()
        self.update_adjacency_lines()
        # print(f"Update Adjacency Lines Time: {time.time() - start_time:.6f}s")

        self.canvas.update()


    def render(self):
        # start = time.time()
        self.update_scene()
        # update_time = time.time() - start

        # start = time.time()
        app.process_events()
        # process_time = time.time() - start

        # print(f"Update Time: {update_time:.6f}s, Process Time: {process_time:.6f}s")





# Mock swarm class for demonstration
# class MockSwarm:
#     class Agent:
#         def __init__(self, position):
#             self.state = position

#     def __init__(self):
#         self.agents = [self.Agent((np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5))) for _ in range(10)]

#     def compute_adjacency_matrix(self):
#         """Create a mock adjacency matrix based on distance."""
#         num_agents = len(self.agents)
#         adjacency_matrix = np.zeros((num_agents, num_agents), dtype=bool)
#         positions = np.array([agent.state[:3] for agent in self.agents])
#         for i in range(num_agents):
#             for j in range(i + 1, num_agents):
#                 distance = np.linalg.norm(positions[i] - positions[j])
#                 if distance < 3.0:  # Connect agents within a threshold distance
#                     adjacency_matrix[i, j] = adjacency_matrix[j, i] = True
#         return adjacency_matrix


# Example usage
# if __name__ == "__main__":
#     swarm = MockSwarm()
#     renderer = SwarmRenderer3D(swarm)

#     # Simulate Gym-like render calls
#     for _ in range(200):
#         swarm.agents = [MockSwarm.Agent((np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5))) for _ in range(30)]
#         renderer.render()
        # time.sleep(0.03)  # Simulate frame delay
