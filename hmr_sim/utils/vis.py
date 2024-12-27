import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class SwarmRenderer:
    def __init__(self, render_type, env, swarm, occupancy_grid, origin, resolution, vis_radius=None, plot_limits=None):
        self.env = env
        self.render_type = render_type
        self.swarm = swarm
        self.occupancy_grid = occupancy_grid
        self.origin = origin
        self.resolution = resolution
        self.vis_radius = vis_radius
        self.fig = None
        self.ax = None
        self.agent_markers = []  # Stores matplotlib Line2D objects (not scatter)
        self.sensor_circles = []
        self.paths = []  # List to store agent paths
        self.old_paths = []  # List to store old agent paths
        self.adjacency_lines = []
        self.battery_circles = []  # List to store battery status circles
        self.type_styles = {
            1: {'cmap': 'Blues', 'marker': 'o'},
            0: {'cmap': 'Reds', 'marker': '^'},
            2: {'cmap': 'YlOrBr', 'marker': 's'}
        }
        self.plot_limits = plot_limits

    def initialize(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

        extent = (
            self.origin['x'],
            self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution,
            self.origin['y'],
            self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution,
        )

        if self.render_type == 'nav':
            self.ax.imshow(1.0 - self.occupancy_grid, cmap='gray', origin='lower', extent=extent, zorder=0)
        elif self.render_type == 'explore':
            # Invert colors: unexplored (-1) remains grey, 0 becomes 1, and 1 becomes 0
            inverted_map = np.where(self.env.exploration_map == -1, 0.5, 1 - self.env.exploration_map)

            # Initial display of the exploration map, mapping unexplored (-1) to a gray color
            self.map_display = self.ax.imshow(
                inverted_map,
                cmap='gray',
                origin='lower',
                extent=extent,
                vmin=0,
                vmax=1,
                zorder=0
            )

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')

        if self.plot_limits is not None:
            self.ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
            self.ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
        else:
            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal', 'box')

        # Initialize paths and old paths
        self.paths = [None] * len(self.swarm.agents)
        self.old_paths = [None] * len(self.swarm.agents)

    def update_markers(self):

        # Remove extra markers if agents are removed
        while len(self.agent_markers) > len(self.swarm.agents):
            marker = self.agent_markers.pop()
            paths = self.paths.pop()
            old_paths = self.old_paths.pop()
            if marker is not None:
                marker.remove()  # Remove the marker from the plot
            if paths is not None:
                paths.remove()
            if old_paths is not None:
                old_paths.remove()

        while len(self.agent_markers) < len(self.swarm.agents):
            self.agent_markers.append(None)
            self.paths.append(None)
            self.old_paths.append(None)

        for i, agent in enumerate(self.swarm.agents):
            style = self.type_styles.get(agent.type, {'cmap': 'Greys', 'marker': 'o'})
            cmap = plt.cm.get_cmap(style['cmap'])
            color = cmap(agent.battery) if agent.battery is not None else mcolors.to_rgb(style['cmap'])

            if self.agent_markers[i] is None:
                marker, = self.ax.plot(
                    [agent.state[0]], [agent.state[1]], style['marker'],
                    mfc=color, mec='black', markersize=15, zorder=3
                )
                self.agent_markers[i] = marker
            else:
                self.agent_markers[i].set_xdata(agent.state[0])
                self.agent_markers[i].set_ydata(agent.state[1])
                self.agent_markers[i].set_markerfacecolor(color)

    def update_adjacency_lines(self):
        for line in self.adjacency_lines:
            line.remove()
        self.adjacency_lines.clear()
        adjacency_matrix = self.swarm.compute_adjacency_matrix()
        positions = [agent.state[:2] for agent in self.swarm.agents]

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if adjacency_matrix[i, j]:
                    line, = self.ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                                         'k-', zorder=1, alpha=0.1, linewidth=2.5)
                    self.adjacency_lines.append(line)

    def update_paths(self):
        for i, agent in enumerate(self.swarm.agents):
            if agent.path is not None:
                path_x = [p[0] for p in agent.path]
                path_y = [p[1] for p in agent.path]

                if self.paths[i]:
                    self.paths[i].set_data(path_x, path_y)
                else:
                    path_line, = self.ax.plot(
                        path_x, path_y, color='green', linestyle='--', linewidth=2, alpha=0.3, zorder=2
                    )
                    self.paths[i] = path_line

    def update_old_paths(self):
        """
        Updates or creates lines for the old paths of each agent, 
        with arrows showing the direction of movement at fixed intervals.
        """

        for i, agent in enumerate(self.swarm.agents):
            if agent.old_path is not None and len(agent.old_path) > 1 and agent.battery > 0.85:
                # Extract x and y coordinates
                old_path_x = [p[0] for p in agent.old_path]
                old_path_y = [p[1] for p in agent.old_path]

                # Update the existing dashed line for the path
                if self.old_paths[i]:
                    self.old_paths[i].set_data(old_path_x, old_path_y)
                else:
                    old_path_line, = self.ax.plot(
                        old_path_x, old_path_y, color='blue', linestyle='--', linewidth=3.0, alpha=0.4, zorder=1
                    )
                    self.old_paths[i] = old_path_line

                if len(agent.old_path) % 6 == 0:
                    dx = old_path_x[-1] - old_path_x[-2]
                    dy = old_path_y[-1] - old_path_y[-2]

                    self.ax.quiver(
                        old_path_x[-2], old_path_y[-2], dx, dy,
                        angles='xy', scale_units='xy', scale=0.6, color='blue', alpha=0.05, zorder=2
                    )

    def update_battery_circles(self):
        # Remove extra circles if agents are removed
        while len(self.battery_circles) > len(self.swarm.agents):
            circle = self.battery_circles.pop()
            if circle is not None:
                circle.remove()  # Remove the circle from the plot

        # Ensure enough circles exist for all agents
        while len(self.battery_circles) < len(self.swarm.agents):
            self.battery_circles.append(None)

        # Update or create battery circles for each agent
        for i, agent in enumerate(self.swarm.agents):
            if agent.battery < self.swarm.add_agent_params['battery_of_concern']:
                color = 'red'
            elif agent.battery is not None and agent.battery > 0.85:
                color = 'green'
            else:
                color = None

            if color:
                if self.battery_circles[i] is None:
                    # Create a new circle
                    circle = plt.Circle(
                        (agent.state[0], agent.state[1]),
                        0.2,
                        edgecolor=color,
                        facecolor='none',
                        linestyle='dashed',
                        linewidth=2,
                        zorder=2,
                        alpha=0.8
                    )
                    self.ax.add_patch(circle)
                    self.battery_circles[i] = circle
                else:
                    # Update the existing circle
                    self.battery_circles[i].center = (agent.state[0], agent.state[1])
                    self.battery_circles[i].set_edgecolor(color)
            else:
                # Remove the circle if it exists and no longer needed
                if self.battery_circles[i]:
                    self.battery_circles[i].remove()
                    self.battery_circles[i] = None

    def update_exploration_map(self):
        # Update the map display with the current inverted exploration map
        inverted_map = np.where(self.env.exploration_map == -1, 0.5, 1 - self.env.exploration_map)
        self.map_display.set_data(inverted_map)

    def render(self):
        if self.fig is None:
            self.initialize()
        self.update_markers()
        self.update_adjacency_lines()
        self.update_paths()
        self.update_old_paths()
        self.update_battery_circles()
        if self.render_type == 'explore':
            self.update_exploration_map()
        plt.draw()
        plt.pause(0.01)
