# encoding: utf-8
# module sgd.py
# from affiliation_graphs
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from lib.common.cupy_support import xp, scatter_add
from lib.bipartite_graph_embeddings.optimisers.optimiser import Optimiser
from math import log
# ---


class SGD(Optimiser):

    """
    Manages optimisation of the Bipartite Graph Embeddings (BGE) technique using Stochastic Gradient Descent (SGD).
    """

    def __init__(self, alpha, num_actors, num_comms, embedding_dim):

        """
        Initialises the SGD class.

        **Parameters**
        > **alpha:** ``float`` -- The initial learning rate.

        > **num_actors:** ``int`` -- The number of vertices in the actor set.

        > **num_comms:** ``int`` -- The number of vertices in the community set.

        > **embedding_dim:** ``int`` -- The number of dimensions of each embedding.

        !!! note "num_actors/num_comms"
            ``num_actors`` and ``num_comms`` are not required for the SGD class; however, they are still defined to keep
            the parameters consistent with the optimiser classes.
        """
        super().__init__(alpha, num_actors, num_comms, embedding_dim)

    def update_timestep(self):
        """Updates the time step 't' and returns the corresponding alpha for this given time step"""
        """Updates the time step 't' and the corresponding alpha for this given time step"""
        self.t += 1
        self.alpha_t = self.alpha / log(self.t)

    def calculate_batch_gradient(self, grad, indices, key):
        """
        Calculates the gradient over a batch using the SGD optimisation technique.

        **Parameters**
        > **grad:** ``array`` -- Individual gradients.

        > **indices:** ``array`` -- The vertices corresponding to the individual gradients.

        > **key:** ``string`` -- Whether or this gradient affects the actor or community set. key='a' if this gradient
        is over the actor set; else key='c'.

        !!! note "key"
            The ``key`` parameter is not required for the SGD class; however, it is still defined here to keep
            the parameters consistent with the optimiser classes.
        """

        # Count the number of changes for each modified vertex.
        unique, counts = xp.unique(indices, return_counts=True)

        # Replace indices with their index in the unique array.
        indices = self.replace_with_unique(indices, unique)

        # Reshape counts to simplify subsequent operations.
        counts = counts.reshape(-1, 1)

        # Create a zero matrix for each vertex to be updated.
        dL = xp.zeros((unique.shape[0], self.embedding_dim), dtype=float)

        # Add all gradients for each respective gradient.
        scatter_add(dL, indices, grad)

        # Calculate the change in the first and second order momentum
        dL = self.alpha_t * xp.divide(dL, counts)

        return unique, dL

