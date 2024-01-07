#!/usr/bin/env python3

"""
This script creates a grid map with obstacles.
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

class OccupancyMap():

    def __init__(self, map_size_x, map_size_y, num_obstacles, start_x, start_y, goal_x, goal_y):
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.num_obstacles = num_obstacles
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.EMPTY_CELL = 0 # Empty cell
        self.OCCUPIED_CELL = 1 # Occupied cell
        self.create_map()

    def create_map(self):
        # Create a map of all empty cells
        self.map_cost = np.zeros(self.map_size_y * self.map_size_x).reshape(self.map_size_y, self.map_size_x)
        # Obstacles
        for i in range(self.num_obstacles):
            self.map_cost[10, i] = self.OCCUPIED_CELL
            self.map_cost[11, i] = self.OCCUPIED_CELL
            self.map_cost[12, i] = self.OCCUPIED_CELL
            self.map_cost[13, i] = self.OCCUPIED_CELL

    def plot_grid(self, x_plot=None):

        # create discrete colormap
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [self.EMPTY_CELL, self.OCCUPIED_CELL, self.OCCUPIED_CELL+1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Plot it out
        fig, ax = plt.subplots()
        # map
        ax.imshow(self.map_cost, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(0.5, self.map_size_y, 1))
        ax.set_yticks(np.arange(0.5, self.map_size_x, 1))
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.text(self.start_y, self.start_x, 'S', ha='center', va='center', color='green')
        ax.text(self.goal_y, self.goal_x, 'G', ha='center', va='center', color='red')
        # trajectory
        if x_plot is not None:
            plt.plot(x_plot[0, 1, :], x_plot[0, 0, :], 'b')
        fig.set_size_inches((8.5, 11), forward=False)
