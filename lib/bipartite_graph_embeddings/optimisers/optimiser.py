# encoding: utf-8
# module optimiser.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from lib.common.cupy_support import xp
# ---


class Optimiser:

    def __init__(self, alpha, num_actors, num_comms, embedding_dim):

        """
        Initialises the Optimiser class.

        **Parameters**
        > **alpha:** ``float`` -- The initial learning rate.

        > **num_actors:** ``int`` -- The number of vertices in the actor set.

        > **num_comms:** ``int`` -- The number of vertices in the community set.

        > **embedding_dim:** ``int`` -- The number of dimensions of each embedding.

        !!! note "num_actors/num_comms"
            ``num_actors`` and ``num_comms`` are not required for the SGD class; however, they are still defined to keep
            the parameters consistent with the optimiser classes.
        """

        # Learning Rate
        self.alpha = alpha

        # Graph parameters
        self.num_actors = num_actors
        self.num_comms = num_comms
        self.embedding_dim = embedding_dim

        # timestep and alpha at the initial timestep.
        self.t = 1
        self.alpha_t = self.alpha

    @staticmethod
    def replace_with_unique(array, unique):
        """
        Replace elements in ``array`` with their index in ``unique``.

        > **array:** ``array`` -- array of elements to replace with their index in unique.

        > **unique:** ``array`` -- A list of unique elements from ``array``.

        **Returns**
        > **unique_array:** ``array`` -- ``array`` where the value of elements have been replace by their index in the
        ``unique`` array.
        """
        sidx = unique.argsort()
        return sidx[xp.searchsorted(unique, array, sorter=sidx)]

    def update_timestep(self):
        """
        Updates the time step 't' and returns the corresponding alpha for this given time step.

        !!! note
            Overridden by child classes.
        """
        pass

    def calculate_batch_gradient(self, grad, indices, key):
        """
        Calculates the gradient over a batch.

        **Parameters**
        > **grad:** ``array`` -- Individual gradients.

        > **indices:** ``array`` -- The vertices corresponding to the individual gradients.

        > **key:** ``string`` -- Whether or this gradient affects the actor or community set. key='a' if this gradient
        is over the actor set; else key='c'.

        !!! note
            Overridden by child classes.
        """
        pass
