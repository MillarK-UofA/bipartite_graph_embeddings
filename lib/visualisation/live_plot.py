# ---
from time import sleep
from sklearn.decomposition import PCA
import math
from lib.common.cupy_support import xp, _cupy_available
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
# ---


class LivePlot:

    marker_info = {
        'R': {'face_colour': '#D18B8A', 'edge_colour': '#700101'},
        'G': {'face_colour': '#ABDEBC', 'edge_colour': '#3A7048'},
        'B': {'face_colour': '#8DAEFF', 'edge_colour': '#15296E'},
        'P': {'face_colour': '#C7A4EB', 'edge_colour': '#493952'},
        'Yl': {'face_colour': '#f7b388', 'edge_colour': '#eb7a34'},
        'Bl': {'face_colour': '#696969', 'edge_colour': '#000000'}
    }

    def __init__(self, subplot, title, plt_style='cartesian', templates=None):
        """Initialises the Plot class."""

        # --- Data --- #
        self.x_values = []
        self.y_values = []
        self.labels = []

        # --- Plot --- #
        self.x_axis_lims = [-1.1, 1.1]
        self.y_axis_lims = [-1.1, 1.1]
        self.axis_lims = self.x_axis_lims + self.y_axis_lims
        self.refresh_time = 1

        # --- Markers --- #
        self.face_colour = []
        self.edge_colour = []
        self.marker_size = 75

        self.plt_style = plt_style
        self.templates = templates

        self.plt = subplot
        self.plt_title = title

    @staticmethod
    def update_axis(new_data, axis):
        """Updates the max/min values for an axis if new max/min values are found."""

        data_max = max(new_data)
        data_min = min(new_data)

        if data_max > axis[1]:
            axis[1] = math.ceil(data_max) + 1

        if data_min < axis[0]:
            axis[0] = math.floor(data_min) - 1

        return axis

    def clear_plt(self):
        # clear plot.
        self.plt.cla()
        self.plt.grid()
        self.plt.set_title(self.plt_title)

        self.plt.axis(self.axis_lims)
        plt.tight_layout()

    def draw_plot(self):
        """Redraws the plot."""

        self.plt.scatter(self.x_values, self.y_values, s=self.marker_size, c=self.face_colour, edgecolors=self.edge_colour)

        if self.plt_style == 'polar':
            # Add circle markers
            for radius in [1, .75, .5, .25, 0]:
                self.plt.annotate(str(radius), (0, radius), textcoords="offset points", xytext=(0, -15), ha='center')
                self.plt.add_patch(plt.Circle((0, 0), radius=radius, fill=False, linestyle='--'))

        if self.labels:
            for i, label in enumerate(self.labels):
                self.plt.annotate(label, (self.x_values[i], self.y_values[i]),
                                  textcoords="offset points", xytext=(0, -15), ha='center')

    def draw_cluster_plot(self, colours):

        # Cluster means.
        clusters, other_x, other_y = self.cluster_mean(colours)

        # Indicate each cluster via an arrow (vector) pointing to the average embedding.
        self.plot_arrows(clusters)

        # Plot the remaining embeddings.
        self.plt.scatter(other_x, other_y, alpha=0.1, s=45, c='k')

    def plot_arrows(self, arrows):

        for arrow in arrows:
            # Plot arrow representing the average embedding for this cluster.
            self.plt.annotate("", xy=(arrow['x'], arrow['y']), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=arrow['colour'], ls='--'))

            # Annotate arrow with information on the cluster.
            x_sign = np.sign(arrow['x'])
            y_sign = np.sign(arrow['y'])
            offset = 20
            self.plt.annotate(arrow['label'], (arrow['x'], arrow['y']), textcoords="offset points",
                xytext=(x_sign * offset, y_sign * offset), color=arrow['colour'], ha='center')

    def update(self, x, y, labels, colours):
        """Updates data."""

        # update data.
        self.x_values = x
        self.y_values = y

        # update axis limits (if required).
        self.x_axis_lims = self.update_axis(self.x_values, self.x_axis_lims)
        self.y_axis_lims = self.update_axis(self.y_values, self.y_axis_lims)
        self.axis_lims = self.x_axis_lims + self.y_axis_lims

        self.labels = labels
        self.face_colour = []
        self.edge_colour = []

        # If no colours specified, set all vertex embeddings to be black.
        if colours is None:
            colours = ['Bl' for _ in range(len(x))]

        for colour in colours:
            self.face_colour.append(self.marker_info[colour]['face_colour'])
            self.edge_colour.append(self.marker_info[colour]['edge_colour'])

        # - Draw Plot ------------------------------------------------------------------------------------------------ #

        self.clear_plt()

        if self.plt_style == 'cluster':
            self.draw_plot()
            self.draw_cluster_plot(colours)
        else:
            self.draw_plot()

        if self.templates:
            self.plot_arrows(self.templates)

        plt.pause(self.refresh_time)
        plt.draw()
        # ------------------------------------------------------------------------------------------------------------ #

    def cluster_mean(self, colours):

        # Get the number of distinct clusters (Represented by their colour).
        #   - Any Black is removed as it denotes 'other'.
        clusters = set(colours)
        clusters.discard('Bl')

        # For each cluster, get the average x and y values.
        cluster_arrows = []
        for cluster in clusters:
            cluster_idx = np.where(np.array(colours) == cluster)
            x = np.mean(self.x_values[cluster_idx])
            y = np.mean(self.y_values[cluster_idx])
            label = "Cluster: {} \n [{:.2f}, {:.2f}] \n (# Vertices {:,})".format(cluster, x, y, len(cluster_idx[0]))

            cluster_arrows.append({
                'x': x,
                'y': y,
                'label': label,
                'colour': self.marker_info[cluster]['edge_colour']
            })

        # For the unknown vertices, return the x and y coordinates.
        other_idx = np.where(np.array(colours) == 'Bl')
        other_x = self.x_values[other_idx]
        other_y = self.y_values[other_idx]

        return cluster_arrows, other_x, other_y

    def save(self, file_name='output'):
        """Saves the plot to a PNG file."""
        self.plt.savefig("lib/graph_analysis/bipartite2vec_proto/{}.png".format(file_name))




