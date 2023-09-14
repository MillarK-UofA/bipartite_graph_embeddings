# encoding: utf-8
# module loss_function.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
import numpy as np

from lib.common.cupy_support import xp, sigmoid
# ---


class LossFunction:

    def __init__(self, activation='sigmoid'):
        """
        Initialises the LossFunction class. All loss functions inherit from this class.

        **Parameters**
        > **activation:** ``string`` -- The activation function to use. Either "sigmoid", "tanh", or "None".
        """

        available_activations = {
            'sigmoid': self.activation_sigmoid,
            'tanh': self.activation_tanh,
            'exp': self.activation_exp,
            'none': self.activation_none
        }

        # Selects which activation function should be used.
        self.activation = available_activations[activation]

    def compute(self, w_a, w_c):
        """
        Computes the change in actor and community weights to optimise the loss function.

        **Parameters**
        > **w_a:** ``array`` -- The actor embeddings to be updated.

        > **w_c:** ``array`` -- The community embeddings to be updated.

        !!! note
            Overridden by child classes.
        """
        pass

    @staticmethod
    def compute_y_true(y_pred, lims=(0, 1)):
        """
        Returns the true matrix.

        **Parameters**
        > **y_pred:** ``array`` -- The prediction matrix. Used to define the shape of the true matrix.

        > **lims:** ``tuple`` -- Defines the positive and negative example values. Default,
        ``negative_exmaples=0`` and ``positive_examples=1``.

        **Returns**
        > **y_true:** ``array`` -- The true matrix.

        """

        s0, _, s2 = y_pred.shape
        y_true = xp.full_like(y_pred, fill_value=lims[0])
        y_true.reshape(s0, -1)[:, ::s2 + 1] = lims[1]

        return y_true

    def activation_sigmoid(self, z):
        """
        Computes the forward and backward passes of the sigmoid activation function.

        **Parameters**
        > **z:** ``array`` -- The input to the sigmoid activation function.

        **Returns**
        > **y_pred:** ``array`` -- The prediction matrix.

        > **y_true:** ``array`` -- The true matrix.

        > **dL_dz:** ``array`` -- The change in loss with respect to the change in z.

        !!! note "Loss"
            Technically the loss should be the squared difference of the true and prediction matrices; however, in
            practice, the difference alone does not significant change the training performance while removing some
            complexity.
        """

        # Compute the elementwise sigmoid of the prediction matrix.
        y_pred = sigmoid(z)

        # Get the true matrix.
        y_true = self.compute_y_true(y_pred, lims=(0, 1))

        # calculate the squared difference between y_true and y_pred matrices.
        loss = xp.subtract(y_true, y_pred)

        # Change in y_pred with respect to z.
        dy_dz = xp.subtract(y_pred, xp.power(y_pred, 2))

        # Calculate the change in loss with respect to the change in z. (Chain Rule).
        dL_dz = xp.multiply(loss, dy_dz)

        return y_pred, y_true, dL_dz

    def activation_exp(self, z):

        diag = xp.identity(z.shape[-1])
        off_diag = xp.subtract(xp.ones_like(diag), diag)

        y_pred = xp.exp(z) * diag + xp.exp(-z) * off_diag

        # Get the true matrix.
        y_true = self.compute_y_true(y_pred, lims=(-np.pi, 0))

        # calculate the difference between y_true and y_pred matrices.
        loss = xp.subtract(y_true, y_pred)

        dy_dz = y_pred * diag - y_pred * off_diag

        dL_dz = xp.multiply(loss, dy_dz)

        return y_pred, y_true, dL_dz

    def activation_tanh(self, z):
        """
        Computes the forward and backward passes of the sigmoid activation function.

        **Parameters**
        > **z:** ``array`` -- The input to the sigmoid activation function.

        **Returns**
        > **y_pred:** ``array`` -- The prediction matrix.

        > **y_true:** ``array`` -- The true matrix.

        > **dL_dz:** ``array`` -- The change in loss with respect to the change in z.

        !!! note "Loss"
            Technically the loss should be the squared difference of the true and prediction matrices; however, in
            practice, the difference alone does not significant change the training performance while removing some
            complexity.
        """

        y_pred = xp.tanh(z)

        # Get the true matrix.
        y_true = self.compute_y_true(y_pred, lims=(xp.tanh(-0.5), xp.tanh(1)))

        # calculate the difference between y_true and y_pred matrices.
        loss = xp.subtract(y_true, y_pred)

        # Change in y_pred with respect to z.
        dy_dz = xp.ones_like(y_pred) - xp.power(xp.tanh(y_pred), 2)

        # Calculate the change in loss with respect to the change in z. (Chain Rule).
        dL_dz = xp.multiply(loss, dy_dz)

        return y_pred, y_true, dL_dz

    def activation_none(self, y_pred):
        """
        No activation function.

        Computes the difference between the true and predicted matrices.

        !!! note "Loss"
            Technically the loss should be the squared difference of the true and prediction matrices; however, in
            practice, the difference alone does not significant change the training performance while removing some
            complexity.

        !!! warning "y_true Configuration"
            The configuration for generating the true matrix for the ``activation_none`` function is setup to work with
            the ``cosine`` loss function. Due to the limits of the cosine function [-1, 1], a negative example is set to
             -1 and a positive samples is set to 1.

        **Parameters**
        > **y_pred:** ``array`` -- The prediction matrix.

        **Returns**
        > **y_pred:** ``array`` -- The prediction matrix.

        > **y_true:** ``array`` -- The true matrix.

        > **loss:** ``array`` -- The difference between the true and predicted matrices.
        """

        # Get the true matrix.
        y_true = self.compute_y_true(y_pred, lims=(-1, 1))

        # calculate the difference between y_true and y_pred matrices.
        dL_dy = xp.subtract(y_true, y_pred)

        return y_pred, y_true, dL_dy

