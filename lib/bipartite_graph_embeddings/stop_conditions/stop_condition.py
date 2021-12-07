# encoding: utf-8
# module stop_condition.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from tqdm import tqdm
import statistics as stats
# ---


class StopCondition:

    """
    Defines the stopping conditions used in Bipartite Graph Embeddings (BGE). Also provides feedback to the user about
    current progress.
    """

    def __init__(self, epoch_sample_size, max_epochs):
        """
        Initialises the StopCondition class.

        **Parameters**
        > **epoch_sample_size:** ``int`` -- The number of epochs to run before checking if the stop condition has been
        met. Averages results of all epochs in the sample size.

        > **max_epochs:** ``int`` or ``None`` -- The number of epochs to run. If ``None``, training is will continue
        until the stop condition is met.
        """

        # Starting Conditions
        self.prev_epoch_loss = 1
        self.delta_epoch_loss = -1
        self.epoch_idx = 0
        self.epoch_loss = []
        self.max_epochs = max_epochs

        # Number of epochs to run before checking stop condition.
        self.epoch_sample_size = epoch_sample_size

        # Progress Bar.
        self.bar = tqdm(total=max_epochs)

        # Stop condition.
        # if max_epochs is set, the stop condition is set to run that many epochs. Else, the stopping condition is set
        # such that training will stop when the loss function does not improve from the previous sample of epochs.
        self.condition = self.max_epoch_condition if max_epochs > 0 else self.improving_loss_condition

    def max_epoch_condition(self):
        """
        Defines the max_epoch stop condition.

        **Returns**
        > **stop_condition:** ``bool`` -- True if the stop condition has not been reached.
        """
        return self.epoch_idx < self.max_epochs

    def improving_loss_condition(self):
        """
        Defines the improving_loss stop condition. Training is stopped when the loss function does not improve over the
        previous ``epoch_sample_size``.

        **Returns**
        > **stop_condition:** ``bool`` -- True if the stop condition has not been reached.
        """
        return self.delta_epoch_loss <= 0

    def update(self, cur_epoch_loss):
        """
        Updates the stopping condition with results from the current epoch.

        **Parameters**
        > **cur_epoch_loss:** ``float`` -- The squared mean error of the current epoch.
        """

        # Record total loss of the current epoch.
        self.epoch_loss.append(cur_epoch_loss)

        self.bar.update(1)
        self.epoch_idx += 1

        if len(self.epoch_loss) == self.epoch_sample_size:

            # Calculate the mean batch loss over the epoch.
            avg_epoch_loss = stats.mean(self.epoch_loss)

            # Calculate the change in loss from the previous epoch.
            self.delta_epoch_loss = avg_epoch_loss - self.prev_epoch_loss

            # Update previous epoch loss with current epoch loss.
            self.prev_epoch_loss = avg_epoch_loss

            # Update bar chart.
            self.bar.set_description("Epoch: {:,} Loss: {:.6f}, delta loss {:.6f}"
                                     .format(self.epoch_idx, avg_epoch_loss, self.delta_epoch_loss))

            # Reset epoch loss for the next sample.
            self.epoch_loss = []

    def check_condition(self):
        """
        Checks whether the stopping condition has been met.

        **Returns**
        > **stop_condition:** ``bool`` -- True if the stop condition has not been reached.
        """
        if self.condition():
            return True
        else:
            self.bar.close()
            return False
