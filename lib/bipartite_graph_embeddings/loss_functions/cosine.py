# encoding: utf-8
# module cosine.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Defines the Cosine Loss Function.

The backwards pass for the cosine loss function is computed as follows:

![cosine_loss_function.png](../../img/cosine_loss_function.PNG)
"""

# ---
from lib.bipartite_graph_embeddings.loss_functions.loss_function import LossFunction
from lib.common.cupy_support import xp
# ---


class Cosine(LossFunction):

    def __init__(self, activation='tanh'):
        """Initialises the Cosine Loss Function class."""
        super().__init__(activation)

    def compute(self, w_a, w_c):
        """
        Computes the change in actor and community weights to optimise cosine loss function.

        **Parameters**
        > **w_a:** ``array`` -- The actor embeddings to be updated.

        > **w_c:** ``array`` -- The community embeddings to be updated.

        **Returns**
        > **dL_dwa:** ``array`` -- The change in loss function with respect to the change in actor embeddings.

        > **dL_dwc:** ``array`` -- The change in loss function with respect to the change in community embeddings.

        > **y_true:** ``array`` -- The true matrix.

        > **y_ture:** `` array`` -- The prediction matrix.
        """

        # Calculate the euclidean norm of the actors and communities
        norm_a = xp.linalg.norm(w_a, axis=2).reshape(w_a.shape[0], 1, w_a.shape[1])
        norm_c = xp.linalg.norm(w_c, axis=2).reshape(w_c.shape[0], 1, w_c.shape[1])

        # Multiply every actors' norm with every community norm.
        norm = xp.matmul(xp.transpose(norm_a, (0, 2, 1)), norm_c)

        # Element wise inverse. Used to divide the actors/communities by every norm.
        norm_inv = xp.power(norm, -1)

        # Calculate the cosine similarity between every actor/community combination.
        cosine_sim = xp.divide(xp.matmul(w_a, xp.transpose(w_c, (0, 2, 1))), norm)

        # Pass the cosine sim through activation function. (forward & backward pass).
        y_pred, y_true, dL_dy = self.activation(cosine_sim)

        # Multiple norm_inv and cosine_sim by the
        norm_inv = xp.multiply(norm_inv, dL_dy)
        cosine_sim = xp.multiply(cosine_sim, dL_dy)

        # - Actors Update -------------------------------------------------------------------------------------------- #

        # Divide the communities by their respective norm.
        d_wa_2 = xp.matmul(norm_inv, w_c)

        # Divide each actor by its l2 squared norm.
        d_wa_3 = xp.divide(w_a, xp.power(xp.transpose(norm_a, (0, 2, 1)), 2))  # Dim [2, 2]

        # Sum the cosine coefficient over the community set
        d_wa_4 = xp.sum(cosine_sim, axis=2).reshape(w_a.shape[0], cosine_sim.shape[1], 1)

        # Multiply each normalised actor by its cosine similarity to the community set.
        d_wa_5 = xp.multiply(d_wa_4, d_wa_3)

        eps = xp.random.rand(*d_wa_2.shape) / 1000

        # Subtract d_wa_3 from d_wa_1
        dL_dwa = d_wa_2 - d_wa_5 + eps
        # ------------------------------------------------------------------------------------------------------------ #

        # - Comms Update --------------------------------------------------------------------------------------------- #

        # Divide the actors by their respective norm.
        d_wc_2 = xp.matmul(xp.transpose(norm_inv, (0, 2, 1)), w_a)

        # Divide each comm by its l2 squared norm.
        d_wc_3 = xp.divide(w_c, xp.power(xp.transpose(norm_c, (0, 2, 1)), 2))  # Dim [2, 2]

        # Sum the cosine coefficient over the actor set.
        d_wc_4 = xp.sum(cosine_sim, axis=1).reshape(w_c.shape[0], cosine_sim.shape[2], 1)

        # Multiply each normalised community by its cosine similarity to the actor set.
        d_wc_5 = xp.multiply(d_wc_4, d_wc_3)

        # Subtract d_wa_3 from d_wa_1
        dL_dwc = d_wc_2 - d_wc_5 + eps

        # ------------------------------------------------------------------------------------------------------------ #

        return dL_dwa, dL_dwc, y_true, y_pred