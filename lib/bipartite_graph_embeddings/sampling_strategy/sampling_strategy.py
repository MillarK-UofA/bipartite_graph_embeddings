# encoding: utf-8
# module sampling_strategy.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

# ---
import random

import h5py as h5py
from lib.common.cupy_support import xp
from lib.common.chunker import chunk_list
import numpy as np
import os
# ---


class SamplingStrategy:

    """
    Defines the parent edge sampling strategy. Contains common functions which inherited by all edge sampling
    strategies.
    """

    # Path to save the created dataset if using large_dataset=True
    temp_dataset_path = "./temp"

    # Chunk size, the number of samples to write to disk at once (only used when large_dataset=True).
    chunk_size = int(10e6)

    def __init__(self, batch_size, ns, large_dataset=True):
        """
        Defines the parent edge sampling class.

        **Parameters**
        > **batch_size:** ``int`` -- The number of positive edges to evaluate at once.

        > **ns:** ``int`` -- The number of negative samples to generate per positive sample.

        > **large_dataset:** ``bool`` -- Whether to store the generated dataset in memory (large_dataset=False) or on
        disk (large_dataset=True).
        """

        self.ns = ns
        self.batch_size = batch_size
        self.dataset = []
        self.sampling_budget = 0
        self.save_path = ""

        # Whether to create a file for the edge list (large_dateset=True) or store it in memory (large_dataset=False).
        self.operator = np if large_dataset else xp
        self.large_dataset = large_dataset

        # Create a dataset folder to store large graph edge lists (large_dataset=True)
        if self.large_dataset and not os.path.exists(self.temp_dataset_path):
            os.mkdir(self.temp_dataset_path)

    def __len__(self):
        """Defines the length of the SamplingStrategy by the length of the generated dataset."""
        return len(self.dataset)

    def generate_corpus(self, graph):
        """
        Evaluates the bipartite graph structure to construct a dataset to train the BGE model. generate_corpus is
        overridden by all child classes.
        """
        pass

    def allocate_dataset(self, shape):
        """
        Evaluates the bipartite graph structure to construct a dataset to train the BGE model. generate_corpus is
        overridden by all child classes.

        !!! Note
            The use of np.uintc as the element type for the dataset variable does restrict the number of nodes that can
            exist in the actor and community sets. However, in its current state, there can be up to 4.3B actors and
            4.3B communities.
        """

        # - Allocate memory to store the dataset --------------------------------------------------------------------- #
        # If large_false=true, store dataset on disk; else, store dataset in memory.
        if not self.large_dataset:
            self.dataset = np.zeros(shape=shape, dtype=np.uintc)
        else:
            #self.dataset = np.memmap(self.save_path, dtype=np.uintc, mode='w+', shape=shape)
            self.dataset = h5py.File(self.save_path, 'w')
            self.dataset.create_dataset('dataset', shape=shape, dtype=np.uintc)
            self.dataset = self.dataset['dataset']

        # ------------------------------------------------------------------------------------------------------------ #

    def shuffle(self):
        """
        shuffles the dataset in chunks.
        """
        for chunk in chunk_list(range(self.sampling_budget), self.chunk_size):
            start = chunk[0]
            end = chunk[-1]+1
            self.dataset[start:end, :] = np.random.permutation(self.dataset[start:end, :])

    def get_batch(self, idx):
        """Gets the next batch of positive and negative samples. get_batch is overridden by all child classes."""
        pass

    @staticmethod
    def get_edge_weight(graph, v, u):
        """
        Returns the edge weight between two vertices. Returns 1 if the graph is unweighted.

        **Parameters**
        > **graph:** ``common.BipartiteGraph`` -- The graph under analysis.

        > **v:** ``str`` -- The first vertex.

        > **u:** ``str`` -- The second vertex.

        **Returns**
        > **weight:** ``int`` -- The weight of the edge. Weight is set to 1 if the graph is unweighted.
        """
        if graph.weighted:
            return graph[v][u]['weight']
        else:
            return 1