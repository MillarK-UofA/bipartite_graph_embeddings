from sklearn.decomposition import PCA
from lib.common.cupy_support import xp, _cupy_available
from lib.visualisation.live_plot import LivePlot
import numpy as np
import matplotlib.pyplot as plt


class BGEPlot:

    def __init__(self, graph, embedding_dim, labels=True, lf='cosine', cluster_plt=True, templates=None):
        """
        Initialises the BGEPlot class.

        **Parameters**
        > **labels:** ``bool`` -- Whether to label the actor and community veritces.

        > **colours:** ``bool`` -- Whether to colour the actor and community vertices.

        > **plt_style:** ``string`` -- Specifies which coordinate systems has been used to represent the embeddings, either
        cartesian (default), or polar.
        """

        # If labels was not specified, set actor and comm labels to None. I.e., labels will not be displayed.
        self.labels = {'a': graph.actors, 'c': graph.comms} if labels else {'a': None, 'c': None}

        # If colours were not specified, set actor and comm embedding colour to None. I.e., grey/black.
        self.colours = graph.colours

        self.weights = {'a': graph.get_weight_dist(actors=True), 'c': graph.get_weight_dist(actors=False)}

        # Set the coordinate system used to represent the embeddings.
        self.plt_style = 'polar' if lf == 'angular' else 'cartesian'

        self.templates = templates if templates else {'a': None, 'c': None}

        self.visualise = False

        self.embeddings_dim = embedding_dim

        if lf != 'angular' and cluster_plt:
            self.plt_style = 'cluster'

        if self.visualise:

            # Change cupy to numpy for matplotlib
            if np != xp:
                self.weights['a'] = xp.asnumpy(self.weights['a'])
                self.weights['c'] = xp.asnumpy(self.weights['c'])

            self.fig, ax = plt.subplots(ncols=2, tight_layout=True)
            self.fig.suptitle('BGE Plot')
            self.actors_plt = LivePlot(ax[0], 'Actors', self.plt_style, self.templates['a'])
            self.comms_plt = LivePlot(ax[1], 'Communities', self.plt_style, self.templates['c'])

        # Current epoch.
        self.epoch_idx = -1

    def cart_to_polar(self, radius, theta):
        x = radius * np.cos(theta).flatten()
        y = radius * np.sin(theta).flatten()
        return x, y

    def get_x_y(self, w_a, w_c):
        """

        :param w_a:
        :param w_c:
        :return:
        """

        if self.plt_style == 'polar' and self.embeddings_dim == 1:
            xa, ya = self.cart_to_polar(self.weights['a'], w_a)
            xc, yc = self.cart_to_polar(self.weights['c'], w_c)
            return xa, ya, xc, yc

        elif self.plt_style == 'cartesian' or self.plt_style == 'cluster':
            if self.embeddings_dim > 2:

                pca = PCA(n_components=2)
                pca.fit(np.concatenate([w_a, w_c]))

                a_coord = pca.transform(w_a)
                c_coord = pca.transform(w_c)

                return a_coord[:, 0], a_coord[:, 1], c_coord[:, 0], c_coord[:, 1]
            else:
                return w_a[:, 0], w_a[:, 1], w_c[:, 0], w_c[:, 1]

    def update(self, W):
        """

        :param W:
        :return:
        """
        self.epoch_idx += 1

        # skip update if embeddings cannot be displayed in a two dimensional embedding space.
        if not self.visualise or self.epoch_idx % 100:
            return

        # Extract embeddings. If using cupy, convert embeddings to numpy arrays before plotting them. Matplotlib does
        # not current work with cupy arrays. While converting the embeddings each time the plot is updated is not ideal,
        # this is only required for visualisation purposes.
        w_a = W['a'] if not _cupy_available else xp.asnumpy(W['a'])
        w_c = W['c'] if not _cupy_available else xp.asnumpy(W['c'])

        # Extract x, y coordinates from embeddings.
        x_a, y_a, x_c, y_c = self.get_x_y(w_a, w_c)

        self.actors_plt.update(x_a, y_a, self.labels['a'], self.colours['a'])
        self.comms_plt.update(x_c, y_c, self.labels['c'], self.colours['c'])

