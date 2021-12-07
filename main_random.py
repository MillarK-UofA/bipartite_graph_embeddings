# encoding: utf-8
# module main_random.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Random biparite graph used to test the scalability of BGE.
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
    **Parameters**
    > **n**: ``int`` -- The number of nodes in the actor set.

    > **m**: ``int`` -- The number of nodes in the community set.

    > **p**: ``float`` -- the probability of edges creation.
    :return:
    """

    m = int(n * (avg_actor_degree/avg_comm_degree))
    p = avg_actor_degree/m

    nx_graph = random_graph(n, m, p)

    graph = BipartiteGraph()
    graph.load_from_edgelist(list(nx_graph.edges))
    graph.print_network_statistics("Random Graph")

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="The input graph path.")
    parser.add_argument("-o", "--output",
                        help="The output embeddings path.")
    parser.add_argument("-d", "--embedding_dim", default=128, type=int,
                        help="Number of embedding dimensions (Default 128).")
    parser.add_argument("-bs", "--batch_size", default=2048, type=int,
                        help="Number of forward/backwards passes to computer concurrently (default 4096).")
    parser.add_argument("-a", "--alpha", default=0.025, type=float,
                        help="The initial learning rate (default 0.025)")
    parser.add_argument("-ns", "--negative_samples", default=5, type=int,
                        help="The number of negative samples for each forward/backward passes (default 5).")
    parser.add_argument("-lf", "--loss_function", default="dot",
                        help="The loss function to use (default 'dot').")
    parser.add_argument("-opt", "--optimiser", default="sgd",
                        help="The optimiser to use (default 'sgd').")
    parser.add_argument("-ss", "--sample_size", default=50, type=int,
                        help="The number of epochs before the stop condition is checked (default 20).")
    parser.add_argument("-wi", "--init", default='normal', type=str,
                        help="Which method to use for initialising weights (default 'normal').")
    parser.add_argument("-s", "--sampling_strategy", default="tes", type=str,
                        help="Which edge sampling strategy to use (default 'tes').")
    parser.add_argument("-n", "--num_actors", default='1e1', type=str,
                        help="The number of epochs before the stop condition is checked (default 20).")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

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

    encoder = BGE(graph, lf=args.loss_function, opt=args.optimiser, embedding_dim=args.embedding_dim,
                  alpha=args.alpha, init=args.init)
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Training BGE ------------------------------------------------------------------------------------------------- #
    print("Training...")
    encoder.train_model(
        batch_size=args.batch_size, ns=args.negative_samples, max_epochs=epochs, sampling=args.sampling_strategy,
        epoch_sample_size=args.sample_size
    )
    # ---------------------------------------------------------------------------------------------------------------- #

    print("dl {} - Actor nodes: {} - {:,} seconds".format(args.sampling_strategy, args.num_actors, int(time() - t1)))
    print("-----\n")

