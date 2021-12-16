# encoding: utf-8
# module transient_edge_sampling.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from lib.bipartite_graph_embeddings.sampling_strategy.sampling_strategy import SamplingStrategy
from lib.common.cupy_support import xp
from math import floor
from time import time
import os
# ---


class TES(SamplingStrategy):

    def __init__(self, graph, batch_size, ns):
        """
        Creates a dataloader object from a given bipartite graph

        **Parameters**
        > **batch_size:** ``int`` -- The number of positive edges to evaluate at once.

        > **ns:** ``int`` -- The number of negative samples to generate per positive sample.

        > **large:** ``bool`` -- Whether to store the generated dataset within GPU memory if available
        (large_dataset=False).
        """

        # Call parent init.
        super().__init__(batch_size, ns)

        # For a static graph, the sampling budget is the number of edges in the graph.
        self.sampling_budget = len(graph.edges())

        # Update batch size. TES evaluates ns+1 edges per update.
        self.batch_size = self.batch_size*(ns+1)

        # The temporary storage of the dataset (if using large_dataset=True)
        self.save_path = os.path.join(self.temp_dataset_path, str(graph) + "_tes.h5")

        # Generate the dataset (positive and negative samples).
        self.generate_corpus(graph)

    def generate_corpus(self, graph):
        """
        Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
        BGE algorithm.

        **Parameters**
        > **graph:** ``common.BipartiteGraph`` -- The graph under analysis.

        **Returns**
        > **edges:** ``array`` -- A 2D array representing of edges (positive and negative).
        """

        # Used to measure the time it takes to generate the dataset.
        start_time = time()

        # allocates memory for the dataset (either on disk or in memory).
        self.allocate_dataset(shape=(self.sampling_budget, 2))

        # - Populate dataset ----------------------------------------------------------------------------------------- #
        for idx, (actor, comm) in enumerate(graph.edges()):
            self.dataset[idx, :] = [graph.actor_idx[actor], graph.comm_idx[comm]]
        # ------------------------------------------------------------------------------------------------------------ #

        # Store dataset on GPU.
        if not self.large_dataset:
            self.dataset = xp.array(self.dataset)

        print("sampling time: {:,} seconds".format(int(time() - start_time)))

    def get_batch(self, idx):
        """
        Gets the next batch.

        **Parameters**
        > **idx:** ``int`` -- the starting index of the current batch.

        **Returns**
        > **actor_batch:** ``np.array`` or ``cp.array`` -- the current actor vertices in the current batch.

        > **comm_batch:** ``np.array`` or ``cp.array`` -- the current community vertices in the current batch.
        """

        # extract batch edges from dataset.
        batch_edges = self.dataset[idx:idx+self.batch_size, :]

        # Get the number of ns+1 chunks that the edge list can be split into.
        num_samples = floor(batch_edges.shape[0] / (self.ns + 1)) * (self.ns + 1)

        # Reshape batch edges to separate the actor and community vertices.
        epoch_edges = xp.array(batch_edges[:num_samples, :].transpose().reshape(2 * (self.ns + 1), -1).transpose())

        return epoch_edges[:, :self.ns + 1], epoch_edges[:, self.ns + 1:]
