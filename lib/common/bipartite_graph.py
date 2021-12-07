# encoding: utf-8
# module bipartite_graphs.py
# from affiliation_graphs
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
Class used to manage all functionality need to store/represent/manage a Bipartite graph.

This class uses cupy if it is available on the system.
"""

# ---
from lib.common.cupy_support import xp, csr_matrix, xp_round
from networkx import Graph
from prettytable import PrettyTable
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
import inspect
# ---


class BipartiteGraph(Graph):
    """
    Class used to manage all functionality need to store/represent/manage a Bipartite graph.

    Inherits from the NetworkX 'Graph' class. The custom functions defined for the Bipartite graph are as follows:

    1. [load_from_edgelist_file](http://localhost:8000/common/bipartite_graph#load_from_edgelist_file)

    2. [load_from_edgelist](http://localhost:8000/common/bipartite_graph#load_from_edgelist)

    3. [_construct_graph](http://localhost:8000/common/bipartite_graph#_construct_graph)

    4. [get_degree_dist](http://localhost:8000/common/bipartite_graph#get_degree_dist)

    5. [print_network_statistics](http://localhost:8000/common/bipartite_graph#print_network_statistics)

    6. [get_incidence_matrix](http://localhost:8000/common/bipartite_graph#get_incidence_matrix)

    7. [generate_corpus](http://localhost:8000/common/bipartite_graph#generate_corpus)

    8. [compute_similarity](http://localhost:8000/common/bipartite_graph#compute_similarity)

    9. [set_vertex_colour](http://localhost:8000/common/bipartite_graph#set_vertex_colour)
    """

    def __init__(self, weighted=False):
        """
        Initialises the BipartiteGraph Class.

        **Parameters**

        > **weighted:** ``bool`` -- Whether the analysis should be conducted on a weighted or unweighted graph.
        Default=False (i.e., unweighted).
        """

        # Initialise the parameters in the parent 'Graph' Class.
        super().__init__()

        self.weighted = weighted

        # A list of all actors and communities contained in the graph.
        # Efficient mapping from index to vertex label. O(1).
        self.actors = []
        self.comms = []

        # A dictionary where key=vertex's label, value=vertex's index.
        # Efficient mapping from vertex label to index. O(1).
        self.actor_idx = {}
        self.comm_idx = {}

        # Actor/Community degree distribution.
        # Used for negative sampling.
        self.actor_degrees = None
        self.comm_degrees = None

        # The density of the graph. (i.e., ratio of existing edges over total possible edges).
        self.density = 0

        # The colour of the vertex embeddings (when displayed on the BGEplot). If None, vertices will be displayed with
        # grey embeddings.
        self.colours = {'a': None, 'c': None}

    def __str__(self):
        return "bg_a{}_c{}".format(len(self.actors), len(self.comms))

    def load_from_edgelist_file(self, fname, dlim='\t', header=False, parse_func=None):
        """
        Formats an edgelist contained in a txt/csv file into an 2D python list. This list is then passed to the function
        load_from_edgelist, which subsequently constructs a graph from the edgelist.

        **Parameters**

        > **fname:** ``str`` -- The path to the edgelist file.

        > **dlim**: ``str`` -- The deliminator between columns in a row. Default = '\t' (tab).

        > **header**: ``bool`` -- Whether the edgelist file contains a header row. Default = False.

        > **weighted:** ``bool`` -- Whether analysis should be conducted on a weighted or unweighted graph. Default =
        False (unweighted).

        !!! warn "Weighted Analysis"
            The script will be aborted if the weighted parameter is set to ``True`` but the edgelist file does not
            contain edges with weights.

        !!! note "File Format"
            The txt/csv file should be in the following format:

            **weighted:** ``actor,community,weight``

            **unweighted:** ``actor,community``

             Note: this example format is using dlim=','.
        """

        # - Try to open file at specified path. ---------------------------------------------------------------------- #
        try:
            edgelist_file = open(fname, 'r', encoding='utf8')
        except FileNotFoundError:
            print("Edge list file could not be located at: {}".format(fname))
            exit()
        except TypeError:
            print("No input file specified")
            exit()
        # ------------------------------------------------------------------------------------------------------------ #

        # - Formatting Edge list ------------------------------------------------------------------------------------- #

        # Load all edges as a list of rows.
        rows = edgelist_file.readlines()

        # Skip first row if the edgelist file contains a header.
        if header:
            rows.pop(0)

        # Default parsing function. Formats each row by:
        #   1. removing new line characters (i.e., '\n')
        #   2. splitting the row by the deliminator.
        def format_row(row):
            return row.strip().split(dlim)

        # If no parsing function is set use default parsing function, 'format_row'.
        if parse_func is None:
            parse_func = format_row

        # Formats each row an stores to a 2D list.
        edgelist = list(map(parse_func, rows))

        # ------------------------------------------------------------------------------------------------------------ #

        # Passes formatted edgelist to function "load_from_edgelist". This allows for the efficient reuse of code.
        self.load_from_edgelist(edgelist)

    def load_from_edgelist(self, edgelist):
        """
        Formats an edglist supplied in a 2D python list into a Pandas dataframe. This Pandas dataframe is then used to
        construct the bipartite graph.

        **Parameters**

        > **edgelist:** ``list`` -- A 2D list of edges. E.g., [[actor_1, community_1, weight], ...].

        > **weighted:** ``bool`` -- Whether analysis should be conducted on a weighted or unweighted graph. Default =
        False (unweighted).
        """

        print("Loading in Edges...")

        # converts edgelist into a Pandas dataframe.
        df_edges = pd.DataFrame(edgelist)
        del edgelist

        # Check if edge list contains weights.
        has_weight = True if df_edges.shape[1] == 3 else False

        # Set column names.
        try:
            df_edges.columns = ['actors', 'communities', 'weight'] if has_weight else ['actors', 'communities']
        except ValueError:
            print("Could not split edges into 2 (unweighted) or 3 (weighted) columns. Please check that your input "
                  "file is formatted correctly or that the correct deliminator is used.")
            exit()

        # Aborts script if edgelist is unweighted but analysis is set to weighted.
        if self.weighted and not has_weight:
            print("ERR: Weighted analysis specified however edges are unweighted.")
            exit()

        # Removes weight column if the edgelist contains weights but analysis is set to unweighted.
        if has_weight and not self.weighted:
            del df_edges['weight']

        # Passes the dataframe to the construction function to build a bipartite graph from the edge list.
        self._construct_graph(df_edges)

    def _construct_graph(self, df_edges):
        """
        Populates the Bipartite graph from a given edgelist (df_edges). The Edgelist must be stored in a Pandas
        dataframe.

        **Parameters**
        > **df_edges:** ``pd.DataFrame`` -- A Pandas dateframe containing the edge list of the desired bipartite graph.

        !!! warn "Indirect Call"
            This function should only be called indirectly through the load_from_edgelist function.
        """

        # Get the set of unique actors and communities in the edgelist.
        self.actors = list(df_edges['actors'].unique())
        self.comms = list(df_edges['communities'].unique())

        # Generate a dictionary mapping the vertex labels to their index.
        self.actor_idx = {actor: idx for idx, actor in enumerate(self.actors)}
        self.comm_idx = {comm: idx for idx, comm in enumerate(self.comms)}

        # Add vertices to the graph such that their index order is preserved.
        self.add_nodes_from(self.actors, bipartite=0)
        self.add_nodes_from(self.comms, bipartite=1)

        # Add edges to the graph.
        if self.weighted:
            df_edges['weight'] = pd.to_numeric(df_edges['weight'])
            self.add_weighted_edges_from(df_edges.values)
        else:
            self.add_edges_from(df_edges.values)

        # Compute graph density
        self.density = bipartite.density(self, self.comms) * 100

    def get_degree_dist(self, vertex_type='actors', pow=1):
        """
        Calculates the degree distribution of either the actor or community set.

        **Parameters**
        > **vertex_type:** ``string`` -- Whether to calulate the degree distribution for the actor set ("actors"),
        community set ("comms"), or all vertices ("all").

        > **pow:** ``float`` -- The power to raise each vertex's degree to. This is useful for offsetting the degree
        distribution as used in Word2Vec (see note below).

        **Returns**
        > **vertices:** ``list`` -- The vertices of the chosen set.

        > **degrees_dist:** ``list`` -- The degree distribution. Note: The order of the degree distribution corresponds
        to the order of the vertices in the 'vertices' variable.

        !!! note "Word2Vec Distribution"
            Setting the 'power' attribute to 3/4 gives you the distribution used in the Word2Vec paper to perform
            negative sampling
        """
        def degree_dist(vertex_idx, vertices, pow):
            degree_dist = xp_round(xp.power(xp.array([degree for _, degree in self.degree(vertices)]), pow), 4)
            degree_dist /= sum(degree_dist)

            vertices = xp.array([vertex_idx[actor] for actor in vertices])

            return vertices, degree_dist

        # If actors is True calculate degree distribution for the actor vertices, else calculate degree distribution for
        # the community vertices.
        if vertex_type == 'actors':
            return degree_dist(self.actor_idx, self.actors, pow)
        elif vertex_type == 'comms':
            return degree_dist(self.comm_idx, self.comms, pow)
        elif vertex_type == 'all':
            actor_vertices, actor_dist = degree_dist(self.actor_idx, self.actors, pow)
            comm_vertices, comm_dist = degree_dist(self.comm_idx, self.comms, pow)
            return xp.concatenate([actor_dist, comm_vertices]), xp.concatenate([actor_dist, comm_dist])
        else:
            "Invalid vertex_type ('{}'). Please select either: 'actors', 'comms', or 'all'."
            exit()

    def get_weight_dist(self, actors=True, pow=1):
        """
        Calculates the weight distribution of either the actor or community set.

        **Parameters**
        > **actors:** ``bool`` -- Whether to calculate the weight distribution over the actor or community set.

        > **pow:** ``float`` -- The power to raise each vertex's degree to. This is useful for offsetting the degree
        distribution as used in Word2Vec (see note below).

        **Returns**
        > **weight_dist:** ``list`` -- The weight distribution of the vertices within either the actor or community set.
        """

        # Select either the actor or community vertices.
        vertices = self.actors if actors else self.comms

        # Check if there are any vertices for this graph. Exit if no vertices exist.
        if not vertices:
            frame = inspect.currentframe()
            func_name = inspect.getframeinfo(frame).function
            print("[Err] No vertices found in graph. Populate the graph before running: {}()".format(func_name))
            exit()

        if self.weighted:

            # Calculate the sum of the weights (raised to a specified power) for each vertex in the selected vertex set.
            weight_dist = [
                sum([xp.power(attr['weight'], pow) for node, attr in self.adj[vertex].items()]) for vertex in vertices
            ]

            weight_dist = xp.array(weight_dist)
            weight_dist = weight_dist / max(weight_dist)

        else:
            weight_dist = xp.ones(len(vertices))

        return weight_dist

    def print_network_statistics(self, title):
        """
        Prints that statistics of the bipartite graph (e.g., number of actor/community vertices). This is useful for
        sanity checking that the graph has been populated correctly.

        **Parameters**
        > **title:** ``str`` -- The title of the table.
        """

        actor_degree_avg = np.mean([degree for _, degree in self.degree(self.actors)])
        comm_degree_avg = np.mean([degree for _, degree in self.degree(self.comms)])

        table = PrettyTable()
        table.title = title
        table.field_names = ["|A|", "|C|", "|E|", "deg|A|", "deg|C|", "density (%)"]
        table.add_row([
            "{:,}".format(len(self.actors)),
            "{:,}".format(len(self.comms)),
            "{:,}".format(len(self.edges)),
            "{:.2f}".format(actor_degree_avg),
            "{:.2f}".format(comm_degree_avg),
            "{:.2e}".format(self.density)
        ])

        print("\n", table, "\n")

    def get_incidence_matrix(self):
        """Returns the [incidence matrix](https://en.wikipedia.org/wiki/Incidence_matrix) of the bipartite graph."""
        return csr_matrix(bipartite.biadjacency_matrix(self, row_order=self.actors, column_order=self.comms, dtype=float))

    def generate_corpus(self):
        """
        Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
        BGE algorithm.

        **Returns**
        > **edges:** ``array`` -- A 2D array representing the edges of the graph.

        > **weights:** ``array`` or ``None`` -- A 1D array representing the weights of the edges within the graph. If
        the graph is unweighted, the weights variable is set to ``None``.
        """
        if self.weighted:

            # For each edge, create an array of edges using the actor_idx, comm_idx.
            edges = xp.array([[self.actor_idx[actor], self.comm_idx[comm]] for actor, comm in self.edges()])

            # For each edge, create an array of weights.
            weights = xp.array([self[actor][comm]['weight'] for actor, comm in self.edges()])

            # Return
            return edges, weights

        else:
            return xp.array([[self.actor_idx[actor], self.comm_idx[comm]] for actor, comm in self.edges()]), None

    def compute_similarity(self, actors=True, norm=True):
        """
        Computes the weighted/unweighted similarity for either the actor or community set.

        **Parameters**
        > **actors:** ``bool`` -- Whether to calculate the similarity between the actor or community set.

        > **norm:** ``bool`` -- ``True`` - calculates the cosine similarity between the specified vertex sets. ``False``
        calculates the dot product between the specified vertex set.

        **Returns**
        > **similarity_matrix:** ``csr_matrix`` -- A matrix representing the similarity of the specified vertex set.
        The similarity matrix is represented in compressed sparse (CSR) row format.
        """

        # Get incidence matrix 'B' of the Bipartite Graph.
        B = self.get_incidence_matrix()

        # If computing similarity between communities (actor=False). Transpose incidence matrix such that communities
        # are represented by the rows in the incidence matrix.
        if not actors:
            B = csr_matrix(B.transpose())

        # Normalise sparse matrix.
        if norm:
            for idx in range(B.shape[0]):
                start_idx, end_idx = B.indptr[idx:idx+2]
                data = B.data[start_idx:end_idx]
                B.data[start_idx:end_idx] = data / np.linalg.norm(data)

        # Return the dot product between the incidence matrix and its transpose.
        return B.dot(B.transpose())

    def set_vertex_colour(self, colour_dict, actor=True):
        """
        Sets what colour the vertex should be represented within the BGE Plot.

        **Parameters**
        > **colour_dict:** ``dict`` --  A dictionary containing a selected colour for each vertex in either the actor
        or community set; where key=actor label, value=colour.

        > E.g., colour_dict = {'a1': 'G', 'a2: 'B'}

        > **actor:** ``bool`` -- Whether to set the colour for the actor or community set.

        !!! note
            A colour for all vertices within the actor or community set must be selected.
        """
        # Selected either the actor or community vertices.
        key, vertices = ('a', self.actors) if actor else ('c', self.comms)

        self.colours[key] = [colour_dict[vertex] for vertex in vertices]

