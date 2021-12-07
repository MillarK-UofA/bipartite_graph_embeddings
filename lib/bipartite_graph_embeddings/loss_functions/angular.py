# encoding: utf-8
# module angular.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Defines the Angular Loss Function.
"""

# ---
from lib.bipartite_graph_embeddings.loss_functions.loss_function import LossFunction
from lib.common.cupy_support import xp
# ---


class Angular(LossFunction):

    def __init__(self, activation='exp'):
        """Initialises the Angular Loss Function class."""
        super().__init__(activation)

    def compute(self, w_a, w_c):
        """
        Computes the change in actor and community weights to optimise the cosine of the difference between to angles.

        **Parameters**
        > **w_a:** ``array`` -- The actor embeddings to be updated.

        > **w_c:** ``array`` -- The community embeddings to be updated.

        **Returns**
        > **dL_dwa:** ``array`` -- The change in loss function with respect to the change in actor embeddings.

        > **dL_dwc:** ``array`` -- The change in loss function with respect to the change in community embeddings.

        > **y_true:** ``array`` -- The true matrix.

        > **y_ture:** `` array`` -- The prediction matrix.
        """

        # Reshape embeddings.
        w_a = w_a.reshape((w_a.shape[0], w_a.shape[2], w_a.shape[1], 1))
        w_c = w_c.reshape((w_c.shape[0], w_c.shape[2], 1, w_c.shape[1]))

        # - Forward Pass --------------------------------------------------------------------------------------------- #
        # Calculate the cosine of the difference between the actor and community embeddings.
        z = xp.cos(xp.subtract(w_a, w_c))

        # Passes z through sigmoid activation function.
        y_pred, y_true, dL_dz = self.activation(z)
        # ------------------------------------------------------------------------------------------------------------ #

        # Calculate change in z with respect to w_a, dz_dwa
        # Note change in z with respect to w_c is simply the negative of dz_dwa.
        dz_dwa = xp.matmul(-xp.sin(w_a), xp.cos(w_c)) + xp.matmul(xp.cos(w_a), xp.sin(w_c))

        dz_dwa = xp.sign(dz_dwa)

        # Calculate change in loss with respect to w_a.
        k = xp.multiply(dL_dz, dz_dwa)

        # Calculate change in loss with respect to w_a.
        dL_dwa = xp.transpose(xp.sum(k, axis=3), axes=(0, 2, 1))

        # Calculate change in loss with respect to w_c.
        dL_dwc = xp.transpose(xp.sum(-k, axis=2), axes=(0, 2, 1))

        return dL_dwa, dL_dwc, y_true, y_pred

