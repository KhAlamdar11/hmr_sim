from vispy import scene, app
import numpy as np
import time

class SwarmRenderer3D:
    def __init__(self, swarm, occupancy_grid=None):
        """
        Initialize the 3D renderer for the swarm and occupancy grid.

        Args:
            swarm: An object containing agents with positions and adjacency matrix.
            occupancy_grid: A dictionary containing 'map' (numpy array), 'res' (float), and 'origin' (list).
        """
        self.swarm = swarm
        self.occupancy_grid = occupancy_grid

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

        # Render the occupancy grid if provided
        if self.occupancy_grid:
            self.render_occupancy_grid()


    def render_occupancy_grid(self):
        """Render the occupancy grid as a static surface at height z=0."""
        occupancy_map = self.occupancy_grid['map']
        resolution = self.occupancy_grid['res']
        origin = self.occupancy_grid['origin']  # Ensure origin is a list or tuple

        # Generate grid points
        rows, cols = occupancy_map.shape
        x = np.linspace(origin['x'], origin['y'] + cols * resolution, cols)
        y = np.linspace(origin['x'], origin['y'] + rows * resolution, rows)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)  # Fixed height at z=0

        # Flatten grid points
        vertices = np.c_[x.ravel(), y.ravel(), z.ravel()]

        # Create grid faces (triangles)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                top_left = i * cols + j
                top_right = top_left + 1
                bottom_left = (i + 1) * cols + j
                bottom_right = bottom_left + 1

                # Two triangles per square
                faces.append([top_left, bottom_left, top_right])
                faces.append([bottom_left, bottom_right, top_right])
        faces = np.array(faces)

        # Map occupancy values to colors
        colors = np.zeros((vertices.shape[0], 4))  # RGBA colors
        for i, value in enumerate(occupancy_map.ravel()):
            if value == 1:  # Occupied
                colors[i] = [0.0, 0.0, 0.0, 1.0]  # Black
            elif value == 0.5:  # Partially occupied
                colors[i] = [0.5, 0.5, 0.5, 1.0]  # Grey
            else:  # Free
                colors[i] = [1.0, 1.0, 1.0, 1.0]  # White

        # Create the mesh for the occupancy grid
        grid_mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, vertex_colors=colors, parent=self.view.scene)




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
        """Efficiently update markers with spheres for different agent types."""
        positions = np.array([agent.state[:3] for agent in self.swarm.agents])
        types = [agent.type for agent in self.swarm.agents]

        # Colors for each type
        type_colors = {
            "UAV": (0.2, 0.6, 1.0, 1.0),  # Blue for UAV
            "Robot": (0.0, 1.0, 0.0, 1.0),  # Green for Robot
            "Base": (1.0, 0.0, 0.0, 1.0),  # Red for Base
        }

        # Ensure we have the same number of spheres as agents
        if len(self.markers) != len(positions):
            # Remove old markers if the count changes
            for marker in self.markers:
                marker.parent = None
            self.markers.clear()

            # Create new spheres for each agent
            for i, position in enumerate(positions):
                sphere = scene.visuals.Sphere(
                    radius=0.3,
                    method='latitude',
                    color=type_colors[types[i]],  # Color based on type
                    shading='smooth',  # Add smooth shading
                    parent=self.view.scene,
                )
                sphere.transform = scene.transforms.MatrixTransform()
                sphere.transform.translate(position)
                self.markers.append(sphere)
        else:
            # Update positions of existing markers
            for i, marker in enumerate(self.markers):
                marker.transform.reset()  # Reset transform
                marker.transform.translate(positions[i])  # Update position
                # Recreate the sphere to update the color (no direct color update in VisPy)
                marker.parent = None  # Remove old marker
                sphere = scene.visuals.Sphere(
                    radius=0.3,
                    method='latitude',
                    color=type_colors[types[i]],  # Update color
                    shading='smooth',
                    parent=self.view.scene,
                )
                sphere.transform = scene.transforms.MatrixTransform()
                sphere.transform.translate(positions[i])
                self.markers[i] = sphere




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
        """Update the markers and adjacency lines."""
        self.update_markers()
        self.update_adjacency_lines()
        self.canvas.update()

    def render(self):
        """Render the scene."""
        self.update_scene()
        app.process_events()

