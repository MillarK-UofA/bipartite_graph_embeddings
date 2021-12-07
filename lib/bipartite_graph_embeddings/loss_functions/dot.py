# encoding: utf-8
# module dot.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Defines the Dot Loss Function.

The backwards pass for the dot loss function is computed as follows:

![dot_loss_function.png](../../img/dot_loss_function.PNG)
"""

# ---
from lib.bipartite_graph_embeddings.loss_functions.loss_function import LossFunction
from lib.common.cupy_support import xp
# ---


class Dot(LossFunction):

    def __init__(self, activation='sigmoid'):
        """Initialises the Dot Loss Function class."""
        super().__init__(activation)

    def compute(self, w_a, w_c):
        """
        Computes the change in actor and community weights to optimise dot loss function.

        **Parameters**
        > **w_a:** ``array`` -- The actor embeddings to be updated.

        > **w_c:** ``array`` -- The community embeddings to be updated.

        **Returns**
        > **dL_dwa:** ``array`` -- The change in loss function with respect to the change in actor embeddings.

        > **dL_dwc:** ``array`` -- The change in loss function with respect to the change in community embeddings.

        > **y_true:** ``array`` -- The true matrix.

        > **y_ture:** `` array`` -- The prediction matrix.
        """

        # Calculates the dot product between all actor and community embeddings.
        z = xp.matmul(w_a, xp.transpose(w_c, axes=(0, 2, 1)))

        # Passes z through sigmoid activation function.
        y_pred, y_true, dL_dz = self.activation(z)

        # Calculates the change in loss with respect to the change in actors.
        dL_dwa = xp.matmul(dL_dz, w_c)

        # Calculates the change in loss with respect to the change in communities.
        dL_dwc = xp.matmul(xp.transpose(dL_dz, (0, 2, 1)), w_a)

        return dL_dwa, dL_dwc, y_true, y_pred


