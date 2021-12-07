# encoding: utf-8
# module main.py
# from bipartite_graph_embeddings
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
The main script used to run Bipartite Graph Embeddings.
"""

# ---
from lib.common.bipartite_graph import BipartiteGraph
from lib.bipartite_graph_embeddings.model import BGE
import argparse
# ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="The path to the input graph.")
    parser.add_argument("-o", "--output",
                        help="The output embeddings path. (Default saves the embeddings into the input folder.")
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
    parser.add_argument("-e", "--epochs", default=-1, type=int,
                        help="The number of iterations to run. (Default (-1) stops training when loss increases.)")

    args = parser.parse_args()

    # If output location not set, saves the embeddings in the same directory/name format as input file.
    if args.output is None:
        args.output = args.input
        args.output = args.output.split('.')[0] + "_bge.txt"

    # - Generate Bipartite Graph from input -------------------------------------------------------------------------- #
    graph = BipartiteGraph(weighted=False)
    graph.load_from_edgelist_file(args.input, dlim='\t')
    graph.print_network_statistics(args.input)
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Initialise BGE using specified parameters. ------------------------------------------------------------------- #
    encoder = BGE(graph, lf=args.loss_function, opt=args.optimiser, embedding_dim=args.embedding_dim,
                  alpha=args.alpha, init=args.init)
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Training BGE ------------------------------------------------------------------------------------------------- #
    encoder.train_model(
        batch_size=args.batch_size, ns=args.negative_samples, max_epochs=args.epochs, sampling=args.sampling_strategy,
        epoch_sample_size=args.sample_size
    )
    # ---------------------------------------------------------------------------------------------------------------- #

    # - Saving Embeddings -------------------------------------------------------------------------------------------- #
    print("\nSaving embeddings...")
    encoder.save_embeddings(args.output)
    print("Embeddings saved to: {}".format(args.output))
    # ---------------------------------------------------------------------------------------------------------------- #
