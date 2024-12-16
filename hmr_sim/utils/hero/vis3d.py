from mpl_toolkits.mplot3d import Axes3D
from hmr_sim.utils.vis import SwarmRenderer
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.colors as mcolors
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3D


class SwarmRenderer3D(SwarmRenderer):
    
    def __init__(self, render_type, env, swarm, occupancy_grid, origin, resolution, vis_radius=None, plot_limits=None):
        super().__init__(render_type, env, swarm, occupancy_grid, origin, resolution, vis_radius, plot_limits)

        self.type_styles = {
            'UAV': {'cmap': 'Blues', 'marker': 'o'},
            'Base': {'cmap': 'Reds', 'marker': '^'},
            'Robot': {'cmap': 'YlOrBr', 'marker': 's'}
        }


    def initialize(self):
        print("Initializing 3D scene...")
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')  # Enable 3D projection

        # Define the extent of the map
        extent = (
            self.origin['x'],
            self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution,
            self.origin['y'],
            self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution,
        )

        # Generate grid for the occupancy grid
        x = np.linspace(self.origin['x'], self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution, self.occupancy_grid.shape[1])
        y = np.linspace(self.origin['y'], self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution, self.occupancy_grid.shape[0])
        x, y = np.meshgrid(x, y)
        z = np.full_like(x, 0.01)  # Arbitrary height value (set to 1.0)

        if self.render_type == 'nav':
            # Display the occupancy grid as a 3D surface
            print("Rendering nav mao,.")
            self.map_surface = self.ax.plot_surface(
                x, y, z, facecolors=plt.cm.gray(1.0 - self.occupancy_grid), rstride=1, cstride=1, antialiased=False, zorder=0
            )
        elif self.render_type == 'explore':
            # Invert colors: unexplored (-1) remains grey, 0 becomes 1, and 1 becomes 0
            inverted_map = np.where(self.env.exploration_map == -1, 0.5, 1 - self.env.exploration_map)
            self.map_surface = self.ax.plot_surface(
                x, y, z, facecolors=plt.cm.gray(inverted_map), rstride=1, cstride=1, antialiased=False, zorder=0
            )

        # Hide axis ticks for cleaner visualization
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.axis('off')

        # Set plot limits
        if self.plot_limits is not None:
            self.ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
            self.ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
            self.ax.set_zlim(0, 2)  # Adjust Z limits if necessary
        else:
            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])
            self.ax.set_zlim(0, 2)  # Default Z limits

        self.ax.set_box_aspect([1, 1, 0.1])  # Adjust aspect ratio (x:y:z)

        # Initialize paths and old paths
        self.paths = [None] * len(self.swarm.agents)
        self.old_paths = [None] * len(self.swarm.agents)

    def create_sphere(self, center, radius, resolution=20):
        """
        Generate data for a sphere centered at `center` with the given `radius`.
        """
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def update_markers(self):
        while len(self.agent_markers) > len(self.swarm.agents):
            marker = self.agent_markers.pop()
            if marker is not None:
                marker.remove()

        while len(self.agent_markers) < len(self.swarm.agents):
            self.agent_markers.append(None)

        for i, agent in enumerate(self.swarm.agents):
            if agent.type == 1:
                # Draw a 3D sphere for agents of type 1
                if self.agent_markers[i] is None or not isinstance(self.agent_markers[i], Poly3DCollection):
                    # Remove existing 2D marker if present
                    if self.agent_markers[i] is not None:
                        self.agent_markers[i].remove()
                    
                    # Create sphere
                    sphere_x, sphere_y, sphere_z = self.create_sphere(agent.state[:3], radius=0.1)
                    sphere = self.ax.plot_surface(
                        sphere_x, sphere_y, sphere_z, color='blue', alpha=0.6, zorder=4
                    )
                    self.agent_markers[i] = sphere
                else:
                    # Update sphere position
                    self.agent_markers[i].remove()
                    sphere_x, sphere_y, sphere_z = self.create_sphere(agent.state[:3], radius=0.1)
                    self.agent_markers[i] = self.ax.plot_surface(
                        sphere_x, sphere_y, sphere_z, color='blue', alpha=0.6, zorder=4
                    )
            else:
                # Draw a 2D marker for other agent types
                style = self.type_styles.get(agent.type, {'cmap': 'Greys', 'marker': 'o'})
                cmap = plt.cm.get_cmap(style['cmap'])
                color = cmap(agent.battery) if agent.battery is not None else mcolors.to_rgb(style['cmap'])

                if self.agent_markers[i] is None or isinstance(self.agent_markers[i], Poly3DCollection):
                    # Remove existing sphere if present
                    if self.agent_markers[i] is not None:
                        self.agent_markers[i].remove()
                    
                    # Create 2D marker
                    marker, = self.ax.plot(
                        [agent.state[0]], [agent.state[1]], [agent.state[2]],  # Include Z-axis
                        style['marker'], mfc=color, mec='black', markersize=10, 
                        zorder=5 if agent.type=='UAV' else 4
                    )
                    self.agent_markers[i] = marker
                else:
                    # Update existing 2D marker position
                    self.agent_markers[i].set_data_3d(
                        [agent.state[0]], [agent.state[1]], [agent.state[2]]
                    )
                    self.agent_markers[i].set_markerfacecolor(color)


    def update_adjacency_lines(self):
        # Remove existing lines
        for line in self.adjacency_lines:
            line.remove()
        self.adjacency_lines.clear()

        # Compute adjacency matrix and positions
        adjacency_matrix = self.swarm.compute_adjacency_matrix()
        positions = [agent.state[:3] for agent in self.swarm.agents]

        # Z-offset to avoid overlap with surface
        z_offset = 0.05

        # Draw adjacency lines
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if adjacency_matrix[i, j]:
                    # Offset Z slightly upwards
                    z1 = pos_i[2] + z_offset
                    z2 = pos_j[2] + z_offset

                    # Create a 3D line
                    line = Line3D(
                        [pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], [z1, z2],
                        color='yellow', alpha=0.3, linewidth=3.0, zorder=3
                    )
                    self.ax.add_line(line)  # Add line to the plot
                    self.adjacency_lines.append(line)




    def update_exploration_map(self):
        if hasattr(self, 'map_surface'):
            self.map_surface.remove()
            del self.map_surface

        inverted_map = np.where(self.env.exploration_map == -1, 0.5, 1 - self.env.exploration_map)
        x = np.linspace(self.origin['x'], self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution,
                        self.occupancy_grid.shape[1])
        y = np.linspace(self.origin['y'], self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution,
                        self.occupancy_grid.shape[0])
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(inverted_map)

        self.map_surface = self.ax.plot_surface(
            x, y, z, facecolors=plt.cm.gray(inverted_map), rstride=1, cstride=1, antialiased=False, zorder=0, alpha=0.8
        )

    def render(self):
        # print(f'render lop start time: {time.time()}')
        if self.fig is None:
            self.initialize()
        # Uncomment only necessary updates
        self.update_markers()
        self.update_adjacency_lines()
        self.update_exploration_map()
        self.fig.canvas.draw_idle()  # Queue partial redraw
        self.fig.canvas.flush_events()  # Process events to allow visualization
        # print(f'render lop end time: {time.time()}')
