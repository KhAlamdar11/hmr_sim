import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.colors as mcolors


class SwarmRenderer:
    def __init__(self, swarm, occupancy_grid, origin, resolution, vis_radius=None):
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
        self.type_styles = {
            0: {'cmap': 'Blues', 'marker': 'o'},
            1: {'cmap': 'Reds', 'marker': '^'},
            2: {'cmap': 'YlOrBr', 'marker': 's'}
        }

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
        self.ax.imshow(1.0 - self.occupancy_grid, cmap='gray', origin='lower', extent=extent, zorder=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.axis('off')
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal', 'box')

        # Initialize paths and old paths
        self.paths = [None] * len(self.swarm.agents)
        self.old_paths = [None] * len(self.swarm.agents)

    def update_markers(self):
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
                    mfc=color, mec='black', markersize=13, zorder=3
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
                                         'k-', zorder=1, alpha=0.1, linewidth=4)
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
        Updates or creates lines for the old paths of each agent.
        """
        for i, agent in enumerate(self.swarm.agents):
            if agent.old_path is not None and len(agent.old_path) > 1:
                print(f"Agent ID: {agent.agent_id}, old_path: {agent.old_path}")
                
                # Extract x and y coordinates
                old_path_x = [p[0] for p in agent.old_path]
                old_path_y = [p[1] for p in agent.old_path]

                if self.old_paths[i]:
                    # Update the existing old path line
                    self.old_paths[i].set_data(old_path_x, old_path_y)
                else:
                    # Create a new old path line
                    old_path_line, = self.ax.plot(
                        old_path_x, old_path_y, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1
                    )
                    self.old_paths[i] = old_path_line

    def render(self):
        if self.fig is None:
            self.initialize()
        self.update_markers()
        self.update_adjacency_lines()
        self.update_paths()
        self.update_old_paths()  # Update old paths during each render
        plt.draw()
        plt.pause(0.01)


def render_exp(self):
    """
    Render the environment with a multi-agent swarm.
    All agents will have the same marker and sensory circle color.
    
    Parameters:
        communication_radius (float, optional): If provided, draws lines connecting agents within this radius.
    """
    if self.fig is None:
        # Initialize figure and axis
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

        # Calculate the extent based on the map configuration
        extent = (
            self.origin['x'],
            self.origin['x'] + self.occupancy_grid.shape[1] * self.resolution,
            self.origin['y'],
            self.origin['y'] + self.occupancy_grid.shape[0] * self.resolution,
        )

        # Invert colors: unexplored (-1) remains grey, 0 becomes 1, and 1 becomes 0
        inverted_map = np.where(self.exploration_map == -1, 0.5, 1 - self.exploration_map)

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

        # Customize the axis for clean visualization
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

        # Initialize agent markers and sensory circles
        self.agent_markers = []
        self.sensor_circles = []

        for agent in self.swarm.agents:
            # Plot agent marker

            if agent.type == 0:
                marker, = self.ax.plot(agent.state[0], agent.state[1], 'o', 
                                        mfc='blue', mec='black', markersize=11, zorder=3)
            elif agent.type == 1:
                marker, = self.ax.plot(agent.state[0], agent.state[1], '^', 
                                        mfc='red', mec='black', markersize=11, zorder=3)
            elif agent.type == 2:
                marker, = self.ax.plot(agent.state[0], agent.state[1], 's', 
                                        mfc='yellow', mec='black', markersize=11, zorder=3)
        
            self.agent_markers.append(marker)

            # Add sensory radius circle
            # sensor_circle = Circle((agent.state[0], agent.state[1]), agent.vis_radius, 
            #                         edgecolor='yellow', facecolor='yellow', linewidth=2.3, alpha=0.03, zorder=3)
            # self.ax.add_patch(sensor_circle)
            # self.sensor_circles.append(sensor_circle)

        # Automatically set plot limits based on the occupancy grid extent
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal', 'box')

        # Store handles for paths and adjacency lines
        self.paths = [[] for _ in self.swarm.agents]
        self.adjacency_lines = []

    else:
        # Update the map display with the current inverted exploration map
        inverted_map = np.where(self.exploration_map == -1, 0.5, 1 - self.exploration_map)
        self.map_display.set_data(inverted_map)

        # Update agent markers and sensory circles
        for i, agent in enumerate(self.swarm.agents):
            self.agent_markers[i].set_xdata(agent.state[0])
            self.agent_markers[i].set_ydata(agent.state[1])
            # self.sensor_circles[i].center = (agent.state[0], agent.state[1])

        # Clear previous adjacency lines
        for line in self.adjacency_lines:
            line.remove()
        self.adjacency_lines.clear()

    # Draw adjacency lines if communication_radius is provided
    if self.vis_radius is not None:
        adjacency_matrix = self.swarm.compute_adjacency_matrix()
        positions = [agent.state[:2] for agent in self.swarm.agents]

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if adjacency_matrix[i, j]:
                    line, = self.ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                                         'k-', zorder=1, alpha=0.1, linewidth=4)
                    self.adjacency_lines.append(line)

    # Draw paths for each agent
    for i, agent in enumerate(self.swarm.agents):
        if agent.path is not None:
            path_x = [p[0] for p in agent.path]
            path_y = [p[1] for p in agent.path]

            if self.paths[i]:
                # Update existing path
                self.paths[i].set_data(path_x, path_y)
            else:
                # Create a new path line
                path_line, = self.ax.plot(path_x, path_y, color='green', linestyle='--', linewidth=2, alpha=0.3, zorder=2)
                self.paths[i] = path_line

    # Incremental update for the plot
    plt.draw()
    plt.pause(0.01) 


