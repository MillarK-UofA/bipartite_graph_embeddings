# encoding: utf-8
# module edge_sampling.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
from tqdm import tqdm
from lib.bipartite_graph_embeddings.sampling_strategy.sampling_strategy import SamplingStrategy
from lib.common.alias_table import alias_setup, alias_draw
from lib.common.cupy_support import xp
import numpy as np
from time import time
import os
# ---


class EdgeSampler(SamplingStrategy):

    def __init__(self, graph, batch_size, ns):
        """Defines the conventional edge sampling strategy."""
        super().__init__(batch_size, ns)

        # The number of edges to evaluate in an epoch.
        self.sampling_budget = 10 * 80 * len(graph)

        # The temporary storage of the dataset (if using large_dataset=True)
        self.save_path = os.path.join(self.temp_dataset_path, str(graph)+"_es.dat")

        # Generate the dataset (positive and negative samples).
        self.generate_corpus(graph)

    def generate_corpus(self, graph):
        """
        Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
        BGE algorithm.

        **Parameters**
        > **graph:** ``common.BipartiteGraph`` -- The graph under analysis.
        """

        # Used to measure the time it takes to generate the dataset.
        start_time = time()

        # allocates memory for the dataset (either on disk or in memory).
        self.allocate_dataset(shape=(self.sampling_budget, self.ns+2))

        # - Define Sampling distributions ---------------------------------------------------------------------------- #

        # Define the distribution of edges relative to their weight metric.
        edge_dist = np.zeros(len(graph.edges))
        for idx, (actor, comm) in enumerate(graph.edges):
            edge_dist[idx] = self.get_edge_weight(graph, actor, comm)

        # Create an alias table for the given edge distribution (positive samples).
        J_pos, q_pos = alias_setup(edge_dist)

        # Create an alias table for the noise distribution (negative samples).
        _, noise_dist = graph.get_degree_dist(vertex_type='comms', pow=3/4)
        J_neg, q_neg = alias_setup(noise_dist)

        # Get the edge list of the graph. Define actor and communities by their relative index in the graph.
        edge_idx = [[graph.actor_idx[actor], graph.comm_idx[comm]] for actor, comm in graph.edges()]

        # ------------------------------------------------------------------------------------------------------------ #

        # - Populate dataset ----------------------------------------------------------------------------------------- #
        for idx in tqdm(range(self.sampling_budget)):

            # draw positive edge (actor, community).
            self.dataset[idx, :2] = edge_idx[alias_draw(J_pos, q_pos)]

            # draw negative community examples.
            for i in range(self.ns):
                self.dataset[idx, 2+i] = alias_draw(J_neg, q_neg)
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
        batch = xp.array(self.dataset[idx:idx+self.batch_size, :])

        return batch[:, :1], batch[:, 1:]

