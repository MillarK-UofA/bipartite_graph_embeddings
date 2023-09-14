# encoding: utf-8
# module initialiser.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from lib.common.cupy_support import xp
from math import sqrt
# ---


class Initialiser:

    def __init__(self, num_actors, num_comms, embedding_dim):
        """
        Initialises the embeddings Initialiser class.

        **Parameters**
        > **num_actors:** ``int`` -- The number of vertices in the actor set.

        > **num_comms:** ``int`` -- The number of vertices in the community set.

        > **embedding_dim:** ``int`` -- The number of dimensions of each embedding.
        """

        # The number of actor vertices.
        self.num_actors = num_actors

        # The number of community vertices.
        self.num_comms = num_comms

        # The number of dimensions of each embedding.
        self.embedding_dim = embedding_dim

        # Options for populating weight initialisation.
        self.init_method = {
            'normal': self.populate_normal,
            'uniform_sqrt': self.populate_uniform_sqrt,
            'polar': self.populate_radial,
            'initial': self.populate_initial,
            'n_sphere': self.populate_n_sphere
        }

    def generate_uniform(self, num, lower=-1.0, upper=1.0):
        """
        Initialises the embeddings from a uniform distribution.

        **Parameters**
        > **lower:** ``int`` or ``float`` -- The lower bound of the uniform distribution.

        > **upper:** ``int`` or ``float`` -- The upper bound of the uniform distribution.

        > **num:** ``int`` -- The number of embeddings to generate.

        **Returns**
        > **embeddings** ``array`` -- 2D array with dimensions [num, embedding_dim].
        """
        return xp.random.uniform(lower, upper, num * self.embedding_dim).reshape(num, self.embedding_dim)

    def generate_normal(self, num, mean=0, std=0.01):
        """
        Initialises the embeddings from a normal distribution.

        **Parameters**
        > **mean:** ``int`` or ``float`` -- The mean value of the normal distribution.

        > **std:** ``int`` or ``float`` -- The standard deviation of the normal distribution.

        > **num:** ``int`` -- The number of embeddings to generate.

        **Returns**
        > **embeddings** ``array`` -- 2D array with dimensions [num, embedding_dim].
        """
        return xp.random.normal(mean, std, num * self.embedding_dim).reshape(num, self.embedding_dim)

    def populate_uniform_sqrt(self):
        """
        Initialises the starting positions for the actor and community vertices. Starting position are chosen from a
        uniform distribution between [-1/sqrt(n), 1/sqrt(n)] where n is the number of vertices in the vertex's set.

        **Returns**
        > **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.
        """

        W = {
            'a': self.generate_uniform(self.num_actors, -1/sqrt(self.num_actors), 1/sqrt(self.num_actors)),
            'c': self.generate_uniform(self.num_comms, -1/sqrt(self.num_comms), 1/sqrt(self.num_comms))
        }

        return W

    def populate_normal(self, mean=0, std=0.01):
        """
        Initialises the starting positions for the actor and community vertices. Starting position are chosen from a
        normal distribution with chosen parameters (mean, std).

        **Parameters**
        > **mean:** ``int`` or ``float`` -- The mean value of the normal distribution.

        > **std:** ``int`` or ``float`` -- The standard deviation of the normal distribution.

        **Returns**
        > **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.
        """

        W = {
            'a': self.generate_normal(self.num_actors, mean, std),
            'c': self.generate_normal(self.num_comms, mean, std)
            #'c': xp.zeros(shape=(self.num_comms, self.embedding_dim))
        }

        return W

    def populate_n_sphere(self):
        """
        Initialises the starting positions for the actor and community vertices. Starting position are chosen from a
        normal distribution with chosen parameters (mean, std).

        **Parameters**
        > **mean:** ``int`` or ``float`` -- The mean value of the normal distribution.

        > **std:** ``int`` or ``float`` -- The standard deviation of the normal distribution.

        **Returns**
        > **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.
        """

        W = {
            'a': self.generate_normal(self.num_actors, 0, 1),
            'c': self.generate_normal(self.num_comms, 0, 1)
        }

        W['a'] = xp.divide(W['a'], xp.linalg.norm(W['a'], ord=2, axis=1).reshape(-1, 1))
        W['c'] = xp.divide(W['c'], xp.linalg.norm(W['c'], ord=2, axis=1).reshape(-1, 1))

        return W

    def populate_radial(self):
        """
        Initialises the starting position for the actor and community vertices. Starting positions are chosen from a
        uniform distribution between -pi and pi.

        **Returns**
        > **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.
        """

        W = {
            'a': self.generate_normal(self.num_actors, 0, xp.pi),
            'c': self.generate_normal(self.num_comms, 0, xp.pi)
        }

        return W

    def populate_initial(self, actor_pos=None, comm_pos=None):

        W = {
            'a': xp.zeros(shape=(self.num_actors, self.embedding_dim)) if actor_pos is None else actor_pos,
            'c': self.generate_uniform(self.num_comms, lower=-1.0, upper=1.0) if comm_pos is None else comm_pos,
        }

        return W

    def populate(self, init):
        """
        Decides which embedding initialisation method to use.

        !!! error "Depreciating"
            This method will likely be depreciated when an optimal initialisation method is determined.
        """
        return self.init_method[init]()



