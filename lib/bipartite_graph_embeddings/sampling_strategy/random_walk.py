# encoding: utf-8
# module random_walk.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
The random walk sampling strategy is heavily based on node2vec.

The code for node2vec has been sourced from GroverA et al. The github repository for node2vec can be
found: https://github.com/aditya-grover/node2vec.

@inproceedings{node2vec-kdd2016,
    author = {Grover, Aditya and Leskovec, Jure},
    title = {node2vec: Scalable Feature Learning for Networks},
    booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    year = {2016}
}
"""


# ---
from tqdm import tqdm
from lib.bipartite_graph_embeddings.sampling_strategy.sampling_strategy import SamplingStrategy
from lib.common.alias_table import alias_setup, alias_draw
from lib.common.chunker import chunk_list
from lib.common.cupy_support import xp
import random
import numpy as np
from time import time
import os
# ---


class RandomWalk(SamplingStrategy):

    def __init__(self, graph, batch_size, ns):
        """Creates a dataloader object from a given bipartite graph"""
        super().__init__(batch_size, ns)

        # - Node2Vec random walk properties -------------------------------------------------------------------------- #
        self.r = 10  # Number of walks per vertex.
        self.l = 80  # Walk length.
        self.p = 1   # Return parameter: the likelihood of immediately revisiting a node in the walk.
        self.q = 1   # In-out parameter: allows the search to differentiate between "inward" and "outward" nodes.
        # ------------------------------------------------------------------------------------------------------------ #

        # The number of edges to evaluate in an epoch.
        self.sampling_budget = 10 * 80 * len(graph)

        # The temporary storage of the dataset (if using large_dataset=True)
        self.save_path = os.path.join(self.temp_dataset_path, str(graph) + "_rw.h5")

        # Generate the dataset (positive and negative samples).
        self.generate_corpus(graph)

    def get_alias_edge(self, graph, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        unnormalized_probs = []
        for dst_nbr in sorted(graph.neighbors(dst)):

            weight = self.get_edge_weight(graph, dst, dst_nbr)

            if dst_nbr == src:
                unnormalized_probs.append(weight/self.p)
            else:
                unnormalized_probs.append(weight/self.q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, graph):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """

        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr].get('weight', 1) for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        for edge in graph.edges():
            # Undirected graph. Add an alias edge for both directions.
            alias_edges[edge] = self.get_alias_edge(graph, edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(graph, edge[1], edge[0])

        return alias_nodes, alias_edges

    def node2vec_walk(self, graph, start_node, alias_nodes, alias_edges):
        """
        Simulate a random walk starting from particular starting node.

        **Parameters**
        > **graph:** ``common.BipartiteGraph`` -- The graph under analysis.

        > **start_node:** ``int`` -- The index of the vertex to start the walk.

        > **walk:** ``list`` -- The list of vertices that comprises the walk.
        """

        walk = [start_node]

        while len(walk) <= self.l:
            cur = walk[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def generate_corpus(self, graph):
        """
        Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
        BGE algorithm.

        **Returns**
        > **edges:** ``array`` -- A 2D array representing the edges of the graph.
        """

        # Used to measure the time it takes to generate the dataset.
        start_time = time()

        # allocates memory for the dataset (either on disk or in memory).
        self.allocate_dataset(shape=(self.sampling_budget, self.ns+2))

        # - Define Sampling distributions ---------------------------------------------------------------------------- #
        # Create an alias table for the noise distribution (negative samples).
        _, noise_dist = graph.get_degree_dist(vertex_type='comms', pow=3/4)
        J_neg, q_neg = alias_setup(noise_dist)

        alias_nodes, alias_edges = self.preprocess_transition_probs(graph)
        # ------------------------------------------------------------------------------------------------------------ #

        # - Populate dataset through a random walk ------------------------------------------------------------------- #

        print("Sampling Dataset.")

        # Initialise progress bar.
        bar = tqdm(total=self.sampling_budget)
        idx = 0

        nodes = list(graph.nodes())
        for walk_iter in range(self.r):

            # Shuffle nodes.
            random.shuffle(nodes)

            for chunk in chunk_list(nodes, max(int(self.chunk_size/self.l), 1)):

                # Create a temporary array to save
                temp = np.zeros(shape=(len(chunk)*self.l, self.ns + 2), dtype=np.uintc)
                chunk_idx = 0

                for node in chunk:

                    # Compute walk
                    walk = self.node2vec_walk(graph, node, alias_nodes, alias_edges)

                    # If the walk starts with an actor or community.
                    offset = 0 if node in graph.actor_idx else 1

                    # For each step in the walk.
                    for step_idx in range(self.l):

                        # draw positive edge (actor, community).
                        temp[chunk_idx, 0] = graph.actor_idx[walk[step_idx + (step_idx + offset) % 2]]
                        temp[chunk_idx, 1] = graph.comm_idx[walk[step_idx + (1 - (step_idx + offset) % 2)]]

                        # draw negative community examples.
                        for i in range(self.ns):
                            temp[chunk_idx, 2 + i] = alias_draw(J_neg, q_neg)

                        chunk_idx += 1
                        bar.update(1)

                self.dataset[idx:idx+chunk_idx] = temp
                idx = idx+chunk_idx
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
