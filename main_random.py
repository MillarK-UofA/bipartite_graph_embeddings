# encoding: utf-8
# module main_random.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Random bipartite graph used to test the scalability of BGE.

Shuffling was only performed once to evaluated the scalability of BGE.
"""

# ---
from lib.common.bipartite_graph import BipartiteGraph
from lib.bipartite_graph_embeddings.model import BGE
from networkx.algorithms.bipartite.generators import random_graph
import argparse
from time import time
# ---


def random_size(n, avg_actor_degree, avg_comm_degree):
    """
    Generates a random bipartite graph based on the number of specified actors and an average degree for both the actor
    and community set.

    **Parameters**
    > **n**: ``int`` -- The number of nodes in the actor set.

    > **avg_actor_degree**: ``float`` -- The average number of communities per actor.

    > **avg_comm_degree**: ``float`` -- The average number of actors per community

    **Returns**
    > **graph**: ``common.BipartiteGraph -- A randomly generated bipartite graph.
    """

    # Calculate the number of communities to generate.
    m = int(n * (avg_actor_degree/avg_comm_degree))

    # Calculate the probability of an edge.
    p = avg_actor_degree/m

    # Generate a random bipartite graph using networkx.
    nx_graph = random_graph(n, m, p)

    # Convert the networkx.Graph into the common.Graph object used in BGE.
    graph = BipartiteGraph()
    graph.load_from_edgelist(list(nx_graph.edges))
    graph.print_network_statistics("Random Graph")

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampling_strategy", default="tes", type=str,
                        help="Which edge sampling strategy to use (default 'tes').")
    parser.add_argument("-n", "--num_actors", default='1e1', type=str,
                        help="The number of epochs before the stop condition is checked (default 20).")
    parser.add_argument("-l", "--large_graph", action='store_true',
                        help="Setting the large_graph flag will evaluate the graph from storage (Slower).")
    args = parser.parse_args()

    # - Generate Bipartite Graph from input -------------------------------------------------------------------------- #

    n = int(float(args.num_actors))

    graph = random_size(n, 205, 5.3)
    sampling_budget = 10 * 80 * len(graph) * 10

    print("-----")

    # sampling_budget = num_random walks * walk_length * num_nodes * epochs.
    if args.sampling_strategy == 'tes':
        epochs = int(sampling_budget / len(graph.edges))
    else:
        epochs = 10

    print("Sampling Budget: {:.1f} M (Epochs: {})".format(sampling_budget/1e6, epochs))
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Initialise BGE using specified parameters. ------------------------------------------------------------------- #
    t1 = time()

    encoder = BGE(graph, lf='dot', opt='sgd', embedding_dim=128, alpha=0.025, init='normal')
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Training BGE ------------------------------------------------------------------------------------------------- #
    encoder.train_model(batch_size=2048, ns=5, max_epochs=epochs,
                        sampling=args.sampling_strategy, epoch_sample_size=epochs, large_graph=args.large_graph)
    # ---------------------------------------------------------------------------------------------------------------- #

    print("dl {} - Actor nodes: {} - {:,} seconds".format(args.sampling_strategy, args.num_actors, int(time() - t1)))
    print("-----\n")

