
Class used to manage all functionality need to store/represent/manage a Bipartite graph.

This class uses cupy if it is available on the system.

______

## **BipartiteGraph**`#!py3 class` { #BipartiteGraph data-toc-label=BipartiteGraph }

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

**class functions & static methods:** 

 - [`__contains__`](#__contains__)
 - [`__getitem__`](#__getitem__)
 - [`__init__`](#__init__)
 - [`__iter__`](#__iter__)
 - [`__len__`](#__len__)
 - [`__str__`](#__str__)
 - [`_construct_graph`](#_construct_graph)
 - [`add_edge`](#add_edge)
 - [`add_edges_from`](#add_edges_from)
 - [`add_node`](#add_node)
 - [`add_nodes_from`](#add_nodes_from)
 - [`add_weighted_edges_from`](#add_weighted_edges_from)
 - [`adjacency`](#adjacency)
 - [`clear`](#clear)
 - [`clear_edges`](#clear_edges)
 - [`compute_similarity`](#compute_similarity)
 - [`copy`](#copy)
 - [`edge_subgraph`](#edge_subgraph)
 - [`generate_corpus`](#generate_corpus)
 - [`get_degree_dist`](#get_degree_dist)
 - [`get_edge_data`](#get_edge_data)
 - [`get_incidence_matrix`](#get_incidence_matrix)
 - [`get_weight_dist`](#get_weight_dist)
 - [`has_edge`](#has_edge)
 - [`has_node`](#has_node)
 - [`is_directed`](#is_directed)
 - [`is_multigraph`](#is_multigraph)
 - [`load_from_edgelist`](#load_from_edgelist)
 - [`load_from_edgelist_file`](#load_from_edgelist_file)
 - [`nbunch_iter`](#nbunch_iter)
 - [`neighbors`](#neighbors)
 - [`number_of_edges`](#number_of_edges)
 - [`number_of_nodes`](#number_of_nodes)
 - [`order`](#order)
 - [`print_network_statistics`](#print_network_statistics)
 - [`remove_edge`](#remove_edge)
 - [`remove_edges_from`](#remove_edges_from)
 - [`remove_node`](#remove_node)
 - [`remove_nodes_from`](#remove_nodes_from)
 - [`set_vertex_colour`](#set_vertex_colour)
 - [`size`](#size)
 - [`subgraph`](#subgraph)
 - [`to_directed`](#to_directed)
 - [`to_directed_class`](#to_directed_class)
 - [`to_undirected`](#to_undirected)
 - [`to_undirected_class`](#to_undirected_class)
 - [`update`](#update)

### *BipartiteGraph*.**__contains__**`#!py3 (self, n)` { #__contains__ data-toc-label=__contains__ }

Returns True if n is a node, False otherwise. Use: 'n in G'.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> 1 in G
True


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __contains__(self, n):
	    
	    try:
	        return n in self._node
	    except TypeError:
	        return False
	
	```

______

### *BipartiteGraph*.**__getitem__**`#!py3 (self, n)` { #__getitem__ data-toc-label=__getitem__ }

Returns a dict of neighbors of node n.  Use: 'G[n]'.

Parameters
----------
n : node
   A node in the graph.

Returns
-------
adj_dict : dictionary
   The adjacency dictionary for nodes connected to n.

Notes
-----
G[n] is the same as G.adj[n] and similar to G.neighbors(n)
(which is an iterator over G.adj[n])

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G[0]
AtlasView({1: {}})


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __getitem__(self, n):
	    
	    return self.adj[n]
	
	```

______

### *BipartiteGraph*.**__init__**`#!py3 (self, weighted=False)` { #__init__ data-toc-label=__init__ }

Initialises the BipartiteGraph Class.

**Parameters**

> **weighted:** ``bool`` -- Whether the analysis should be conducted on a weighted or unweighted graph.
Default=False (i.e., unweighted).


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, weighted=False):
	    
	
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
	
	```

______

### *BipartiteGraph*.**__iter__**`#!py3 (self)` { #__iter__ data-toc-label=__iter__ }

Iterate over the nodes. Use: 'for n in G'.

Returns
-------
niter : iterator
    An iterator over all nodes in the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> [n for n in G]
[0, 1, 2, 3]
>>> list(G)
[0, 1, 2, 3]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __iter__(self):
	    
	    return iter(self._node)
	
	```

______

### *BipartiteGraph*.**__len__**`#!py3 (self)` { #__len__ data-toc-label=__len__ }

Returns the number of nodes in the graph. Use: 'len(G)'.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
number_of_nodes: identical method
order: identical method

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> len(G)
4


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __len__(self):
	    
	    return len(self._node)
	
	```

______

### *BipartiteGraph*.**__str__**`#!py3 (self)` { #__str__ data-toc-label=__str__ }

Returns a short summary of the graph.

Returns
-------
info : string
    Graph information as provided by `nx.info`

Examples
--------
>>> G = nx.Graph(name="foo")
>>> str(G)
"Graph named 'foo' with 0 nodes and 0 edges"

>>> G = nx.path_graph(3)
>>> str(G)
'Graph with 3 nodes and 2 edges'


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __str__(self):
	    return "bg_a{}_c{}".format(len(self.actors), len(self.comms))
	
	```

______

### *BipartiteGraph*.**_construct_graph**`#!py3 (self, df_edges)` { #_construct_graph data-toc-label=_construct_graph }

Populates the Bipartite graph from a given edgelist (df_edges). The Edgelist must be stored in a Pandas
dataframe.

**Parameters**
> **df_edges:** ``pd.DataFrame`` -- A Pandas dateframe containing the edge list of the desired bipartite graph.

!!! warn "Indirect Call"
    This function should only be called indirectly through the load_from_edgelist function.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def _construct_graph(self, df_edges):
	    
	
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
	
	```

______

### *BipartiteGraph*.**add_edge**`#!py3 (self, u_of_edge, v_of_edge, **attr)` { #add_edge data-toc-label=add_edge }

Add an edge between u and v.

The nodes u and v will be automatically added if they are
not already in the graph.

Edge attributes can be specified with keywords or by directly
accessing the edge's attribute dictionary. See examples below.

Parameters
----------
u_of_edge, v_of_edge : nodes
    Nodes can be, for example, strings or numbers.
    Nodes must be hashable (and not None) Python objects.
attr : keyword arguments, optional
    Edge data (or labels or objects) can be assigned using
    keyword arguments.

See Also
--------
add_edges_from : add a collection of edges

Notes
-----
Adding an edge that already exists updates the edge data.

Many NetworkX algorithms designed for weighted graphs use
an edge attribute (by default `weight`) to hold a numerical value.

Examples
--------
The following all add the edge e=(1, 2) to graph G:

>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> e = (1, 2)
>>> G.add_edge(1, 2)  # explicit two-node form
>>> G.add_edge(*e)  # single edge as tuple of two nodes
>>> G.add_edges_from([(1, 2)])  # add edges from iterable container

Associate data to edges using keywords:

>>> G.add_edge(1, 2, weight=3)
>>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

For non-string attribute keys, use subscript notation.

>>> G.add_edge(1, 2)
>>> G[1][2].update({0: 5})
>>> G.edges[1, 2].update({0: 5})


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def add_edge(self, u_of_edge, v_of_edge, **attr):
	    
	    u, v = u_of_edge, v_of_edge
	    # add nodes
	    if u not in self._node:
	        if u is None:
	            raise ValueError("None cannot be a node")
	        self._adj[u] = self.adjlist_inner_dict_factory()
	        self._node[u] = self.node_attr_dict_factory()
	    if v not in self._node:
	        if v is None:
	            raise ValueError("None cannot be a node")
	        self._adj[v] = self.adjlist_inner_dict_factory()
	        self._node[v] = self.node_attr_dict_factory()
	    # add the edge
	    datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
	    datadict.update(attr)
	    self._adj[u][v] = datadict
	    self._adj[v][u] = datadict
	
	```

______

### *BipartiteGraph*.**add_edges_from**`#!py3 (self, ebunch_to_add, **attr)` { #add_edges_from data-toc-label=add_edges_from }

Add all the edges in ebunch_to_add.

Parameters
----------
ebunch_to_add : container of edges
    Each edge given in the container will be added to the
    graph. The edges must be given as 2-tuples (u, v) or
    3-tuples (u, v, d) where d is a dictionary containing edge data.
attr : keyword arguments, optional
    Edge data (or labels or objects) can be assigned using
    keyword arguments.

See Also
--------
add_edge : add a single edge
add_weighted_edges_from : convenient way to add weighted edges

Notes
-----
Adding the same edge twice has no effect but any edge data
will be updated when each duplicate edge is added.

Edge attributes specified in an ebunch take precedence over
attributes specified via keyword arguments.

Examples
--------
>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
>>> e = zip(range(0, 3), range(1, 4))
>>> G.add_edges_from(e)  # Add the path graph 0-1-2-3

Associate data to edges

>>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
>>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def add_edges_from(self, ebunch_to_add, **attr):
	    
	    for e in ebunch_to_add:
	        ne = len(e)
	        if ne == 3:
	            u, v, dd = e
	        elif ne == 2:
	            u, v = e
	            dd = {}  # doesn't need edge_attr_dict_factory
	        else:
	            raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
	        if u not in self._node:
	            if u is None:
	                raise ValueError("None cannot be a node")
	            self._adj[u] = self.adjlist_inner_dict_factory()
	            self._node[u] = self.node_attr_dict_factory()
	        if v not in self._node:
	            if v is None:
	                raise ValueError("None cannot be a node")
	            self._adj[v] = self.adjlist_inner_dict_factory()
	            self._node[v] = self.node_attr_dict_factory()
	        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
	        datadict.update(attr)
	        datadict.update(dd)
	        self._adj[u][v] = datadict
	        self._adj[v][u] = datadict
	
	```

______

### *BipartiteGraph*.**add_node**`#!py3 (self, node_for_adding, **attr)` { #add_node data-toc-label=add_node }

Add a single node `node_for_adding` and update node attributes.

Parameters
----------
node_for_adding : node
    A node can be any hashable Python object except None.
attr : keyword arguments, optional
    Set or change node attributes using key=value.

See Also
--------
add_nodes_from

Examples
--------
>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_node(1)
>>> G.add_node("Hello")
>>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
>>> G.add_node(K3)
>>> G.number_of_nodes()
3

Use keywords set/change node attributes:

>>> G.add_node(1, size=10)
>>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))

Notes
-----
A hashable object is one that can be used as a key in a Python
dictionary. This includes strings, numbers, tuples of strings
and numbers, etc.

On many platforms hashable items also include mutables such as
NetworkX Graphs, though one should be careful that the hash
doesn't change on mutables.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def add_node(self, node_for_adding, **attr):
	    
	    if node_for_adding not in self._node:
	        if node_for_adding is None:
	            raise ValueError("None cannot be a node")
	        self._adj[node_for_adding] = self.adjlist_inner_dict_factory()
	        attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
	        attr_dict.update(attr)
	    else:  # update attr even if node already exists
	        self._node[node_for_adding].update(attr)
	
	```

______

### *BipartiteGraph*.**add_nodes_from**`#!py3 (self, nodes_for_adding, **attr)` { #add_nodes_from data-toc-label=add_nodes_from }

Add multiple nodes.

Parameters
----------
nodes_for_adding : iterable container
    A container of nodes (list, dict, set, etc.).
    OR
    A container of (node, attribute dict) tuples.
    Node attributes are updated using the attribute dict.
attr : keyword arguments, optional (default= no attributes)
    Update attributes for all nodes in nodes.
    Node attributes specified in nodes as a tuple take
    precedence over attributes specified via keyword arguments.

See Also
--------
add_node

Examples
--------
>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_nodes_from("Hello")
>>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
>>> G.add_nodes_from(K3)
>>> sorted(G.nodes(), key=str)
[0, 1, 2, 'H', 'e', 'l', 'o']

Use keywords to update specific node attributes for every node.

>>> G.add_nodes_from([1, 2], size=10)
>>> G.add_nodes_from([3, 4], weight=0.4)

Use (node, attrdict) tuples to update attributes for specific nodes.

>>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
>>> G.nodes[1]["size"]
11
>>> H = nx.Graph()
>>> H.add_nodes_from(G.nodes(data=True))
>>> H.nodes[1]["size"]
11


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def add_nodes_from(self, nodes_for_adding, **attr):
	    
	    for n in nodes_for_adding:
	        try:
	            newnode = n not in self._node
	            newdict = attr
	        except TypeError:
	            n, ndict = n
	            newnode = n not in self._node
	            newdict = attr.copy()
	            newdict.update(ndict)
	        if newnode:
	            if n is None:
	                raise ValueError("None cannot be a node")
	            self._adj[n] = self.adjlist_inner_dict_factory()
	            self._node[n] = self.node_attr_dict_factory()
	        self._node[n].update(newdict)
	
	```

______

### *BipartiteGraph*.**add_weighted_edges_from**`#!py3 (self, ebunch_to_add, weight='weight', **attr)` { #add_weighted_edges_from data-toc-label=add_weighted_edges_from }

Add weighted edges in `ebunch_to_add` with specified weight attr

Parameters
----------
ebunch_to_add : container of edges
    Each edge given in the list or container will be added
    to the graph. The edges must be given as 3-tuples (u, v, w)
    where w is a number.
weight : string, optional (default= 'weight')
    The attribute name for the edge weights to be added.
attr : keyword arguments, optional (default= no attributes)
    Edge attributes to add/update for all edges.

See Also
--------
add_edge : add a single edge
add_edges_from : add multiple edges

Notes
-----
Adding the same edge twice for Graph/DiGraph simply updates
the edge data. For MultiGraph/MultiDiGraph, duplicate edges
are stored.

Examples
--------
>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
	    
	    self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)
	
	```

______

### *BipartiteGraph*.**adjacency**`#!py3 (self)` { #adjacency data-toc-label=adjacency }

Returns an iterator over (node, adjacency dict) tuples for all nodes.

For directed graphs, only outgoing neighbors/adjacencies are included.

Returns
-------
adj_iter : iterator
   An iterator over (node, adjacency dictionary) for all nodes in
   the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
[(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def adjacency(self):
	    
	    return iter(self._adj.items())
	
	```

______

### *BipartiteGraph*.**clear**`#!py3 (self)` { #clear data-toc-label=clear }

Remove all nodes and edges from the graph.

This also removes the name, and all graph, node, and edge attributes.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.clear()
>>> list(G.nodes)
[]
>>> list(G.edges)
[]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def clear(self):
	    
	    self._adj.clear()
	    self._node.clear()
	    self.graph.clear()
	
	```

______

### *BipartiteGraph*.**clear_edges**`#!py3 (self)` { #clear_edges data-toc-label=clear_edges }

Remove all edges from the graph without altering nodes.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.clear_edges()
>>> list(G.nodes)
[0, 1, 2, 3]
>>> list(G.edges)
[]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def clear_edges(self):
	    
	    for neighbours_dict in self._adj.values():
	        neighbours_dict.clear()
	
	```

______

### *BipartiteGraph*.**compute_similarity**`#!py3 (self, actors=True, norm=True)` { #compute_similarity data-toc-label=compute_similarity }

Computes the weighted/unweighted similarity for either the actor or community set.

**Parameters**
> **actors:** ``bool`` -- Whether to calculate the similarity between the actor or community set.

> **norm:** ``bool`` -- ``True`` - calculates the cosine similarity between the specified vertex sets. ``False``
calculates the dot product between the specified vertex set.

**Returns**
> **similarity_matrix:** ``csr_matrix`` -- A matrix representing the similarity of the specified vertex set.
The similarity matrix is represented in compressed sparse (CSR) row format.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def compute_similarity(self, actors=True, norm=True):
	    
	
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
	
	```

______

### *BipartiteGraph*.**copy**`#!py3 (self, as_view=False)` { #copy data-toc-label=copy }

Returns a copy of the graph.

The copy method by default returns an independent shallow copy
of the graph and attributes. That is, if an attribute is a
container, that container is shared by the original an the copy.
Use Python's `copy.deepcopy` for new containers.

If `as_view` is True then a view is returned instead of a copy.

Notes
-----
All copies reproduce the graph structure, but data attributes
may be handled in different ways. There are four types of copies
of a graph that people might want.

Deepcopy -- A "deepcopy" copies the graph structure as well as
all data attributes and any objects they might contain.
The entire graph object is new so that changes in the copy
do not affect the original object. (see Python's copy.deepcopy)

Data Reference (Shallow) -- For a shallow copy the graph structure
is copied but the edge, node and graph attribute dicts are
references to those in the original graph. This saves
time and memory but could cause confusion if you change an attribute
in one graph and it changes the attribute in the other.
NetworkX does not provide this level of shallow copy.

Independent Shallow -- This copy creates new independent attribute
dicts and then does a shallow copy of the attributes. That is, any
attributes that are containers are shared between the new graph
and the original. This is exactly what `dict.copy()` provides.
You can obtain this style copy using:

    >>> G = nx.path_graph(5)
    >>> H = G.copy()
    >>> H = G.copy(as_view=False)
    >>> H = nx.Graph(G)
    >>> H = G.__class__(G)

Fresh Data -- For fresh data, the graph structure is copied while
new empty data attribute dicts are created. The resulting graph
is independent of the original and it has no edge, node or graph
attributes. Fresh copies are not enabled. Instead use:

    >>> H = G.__class__()
    >>> H.add_nodes_from(G)
    >>> H.add_edges_from(G.edges)

View -- Inspired by dict-views, graph-views act like read-only
versions of the original graph, providing a copy of the original
structure without requiring any memory for copying the information.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/3/library/copy.html.

Parameters
----------
as_view : bool, optional (default=False)
    If True, the returned graph-view provides a read-only view
    of the original graph without actually copying any data.

Returns
-------
G : Graph
    A copy of the graph.

See Also
--------
to_directed: return a directed copy of the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> H = G.copy()


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def copy(self, as_view=False):
	    
	    if as_view is True:
	        return nx.graphviews.generic_graph_view(self)
	    G = self.__class__()
	    G.graph.update(self.graph)
	    G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
	    G.add_edges_from(
	        (u, v, datadict.copy())
	        for u, nbrs in self._adj.items()
	        for v, datadict in nbrs.items()
	    )
	    return G
	
	```

______

### *BipartiteGraph*.**edge_subgraph**`#!py3 (self, edges)` { #edge_subgraph data-toc-label=edge_subgraph }

Returns the subgraph induced by the specified edges.

The induced subgraph contains each edge in `edges` and each
node incident to any one of those edges.

Parameters
----------
edges : iterable
    An iterable of edges in this graph.

Returns
-------
G : Graph
    An edge-induced subgraph of this graph with the same edge
    attributes.

Notes
-----
The graph, edge, and node attributes in the returned subgraph
view are references to the corresponding attributes in the original
graph. The view is read-only.

To create a full graph version of the subgraph with its own copy
of the edge or node attributes, use::

    G.edge_subgraph(edges).copy()

Examples
--------
>>> G = nx.path_graph(5)
>>> H = G.edge_subgraph([(0, 1), (3, 4)])
>>> list(H.nodes)
[0, 1, 3, 4]
>>> list(H.edges)
[(0, 1), (3, 4)]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def edge_subgraph(self, edges):
	    
	    return nx.edge_subgraph(self, edges)
	
	```

______

### *BipartiteGraph*.**generate_corpus**`#!py3 (self)` { #generate_corpus data-toc-label=generate_corpus }

Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
BGE algorithm.

**Returns**
> **edges:** ``array`` -- A 2D array representing the edges of the graph.

> **weights:** ``array`` or ``None`` -- A 1D array representing the weights of the edges within the graph. If
the graph is unweighted, the weights variable is set to ``None``.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_corpus(self):
	    
	    if self.weighted:
	
	        # For each edge, create an array of edges using the actor_idx, comm_idx.
	        edges = xp.array([[self.actor_idx[actor], self.comm_idx[comm]] for actor, comm in self.edges()])
	
	        # For each edge, create an array of weights.
	        weights = xp.array([self[actor][comm]['weight'] for actor, comm in self.edges()])
	
	        # Return
	        return edges, weights
	
	    else:
	        return xp.array([[self.actor_idx[actor], self.comm_idx[comm]] for actor, comm in self.edges()]), None
	
	```

______

### *BipartiteGraph*.**get_degree_dist**`#!py3 (self, vertex_type='actors', pow=1)` { #get_degree_dist data-toc-label=get_degree_dist }

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


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_degree_dist(self, vertex_type='actors', pow=1):
	    
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
	
	```

______

### *BipartiteGraph*.**get_edge_data**`#!py3 (self, u, v, default=None)` { #get_edge_data data-toc-label=get_edge_data }

Returns the attribute dictionary associated with edge (u, v).

This is identical to `G[u][v]` except the default is returned
instead of an exception if the edge doesn't exist.

Parameters
----------
u, v : nodes
default:  any Python object (default=None)
    Value to return if the edge (u, v) is not found.

Returns
-------
edge_dict : dictionary
    The edge attribute dictionary.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G[0][1]
{}

Warning: Assigning to `G[u][v]` is not permitted.
But it is safe to assign attributes `G[u][v]['foo']`

>>> G[0][1]["weight"] = 7
>>> G[0][1]["weight"]
7
>>> G[1][0]["weight"]
7

>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.get_edge_data(0, 1)  # default edge data is {}
{}
>>> e = (0, 1)
>>> G.get_edge_data(*e)  # tuple form
{}
>>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
0


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_edge_data(self, u, v, default=None):
	    
	    try:
	        return self._adj[u][v]
	    except KeyError:
	        return default
	
	```

______

### *BipartiteGraph*.**get_incidence_matrix**`#!py3 (self)` { #get_incidence_matrix data-toc-label=get_incidence_matrix }

Returns the [incidence matrix](https://en.wikipedia.org/wiki/Incidence_matrix) of the bipartite graph.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_incidence_matrix(self):
	    
	    return csr_matrix(bipartite.biadjacency_matrix(self, row_order=self.actors, column_order=self.comms, dtype=float))
	
	```

______

### *BipartiteGraph*.**get_weight_dist**`#!py3 (self, actors=True, pow=1)` { #get_weight_dist data-toc-label=get_weight_dist }

Calculates the weight distribution of either the actor or community set.

**Parameters**
> **actors:** ``bool`` -- Whether to calculate the weight distribution over the actor or community set.

> **pow:** ``float`` -- The power to raise each vertex's degree to. This is useful for offsetting the degree
distribution as used in Word2Vec (see note below).

**Returns**
> **weight_dist:** ``list`` -- The weight distribution of the vertices within either the actor or community set.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_weight_dist(self, actors=True, pow=1):
	    
	
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
	
	```

______

### *BipartiteGraph*.**has_edge**`#!py3 (self, u, v)` { #has_edge data-toc-label=has_edge }

Returns True if the edge (u, v) is in the graph.

This is the same as `v in G[u]` without KeyError exceptions.

Parameters
----------
u, v : nodes
    Nodes can be, for example, strings or numbers.
    Nodes must be hashable (and not None) Python objects.

Returns
-------
edge_ind : bool
    True if edge is in the graph, False otherwise.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.has_edge(0, 1)  # using two nodes
True
>>> e = (0, 1)
>>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
True
>>> e = (0, 1, {"weight": 7})
>>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
True

The following syntax are equivalent:

>>> G.has_edge(0, 1)
True
>>> 1 in G[0]  # though this gives KeyError if 0 not in G
True


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def has_edge(self, u, v):
	    
	    try:
	        return v in self._adj[u]
	    except KeyError:
	        return False
	
	```

______

### *BipartiteGraph*.**has_node**`#!py3 (self, n)` { #has_node data-toc-label=has_node }

Returns True if the graph contains the node n.

Identical to `n in G`

Parameters
----------
n : node

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.has_node(0)
True

It is more readable and simpler to use

>>> 0 in G
True


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def has_node(self, n):
	    
	    try:
	        return n in self._node
	    except TypeError:
	        return False
	
	```

______

### *BipartiteGraph*.**is_directed**`#!py3 (self)` { #is_directed data-toc-label=is_directed }

Returns True if graph is directed, False otherwise.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def is_directed(self):
	    
	    return False
	
	```

______

### *BipartiteGraph*.**is_multigraph**`#!py3 (self)` { #is_multigraph data-toc-label=is_multigraph }

Returns True if graph is a multigraph, False otherwise.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def is_multigraph(self):
	    
	    return False
	
	```

______

### *BipartiteGraph*.**load_from_edgelist**`#!py3 (self, edgelist)` { #load_from_edgelist data-toc-label=load_from_edgelist }

Formats an edglist supplied in a 2D python list into a Pandas dataframe. This Pandas dataframe is then used to
construct the bipartite graph.

**Parameters**

> **edgelist:** ``list`` -- A 2D list of edges. E.g., [[actor_1, community_1, weight], ...].

> **weighted:** ``bool`` -- Whether analysis should be conducted on a weighted or unweighted graph. Default =
False (unweighted).


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def load_from_edgelist(self, edgelist):
	    
	
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
	
	```

______

### *BipartiteGraph*.**load_from_edgelist_file**`#!py3 (self, fname, dlim='\t', header=False, parse_func=None)` { #load_from_edgelist_file data-toc-label=load_from_edgelist_file }

Formats an edgelist contained in a txt/csv file into an 2D python list. This list is then passed to the function
load_from_edgelist, which subsequently constructs a graph from the edgelist.

**Parameters**

> **fname:** ``str`` -- The path to the edgelist file.

> **dlim**: ``str`` -- The deliminator between columns in a row. Default = '    ' (tab).

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


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def load_from_edgelist_file(self, fname, dlim='\t', header=False, parse_func=None):
	    
	
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
	
	```

______

### *BipartiteGraph*.**nbunch_iter**`#!py3 (self, nbunch=None)` { #nbunch_iter data-toc-label=nbunch_iter }

Returns an iterator over nodes contained in nbunch that are
also in the graph.

The nodes in nbunch are checked for membership in the graph
and if not are silently ignored.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
    The view will only report edges incident to these nodes.

Returns
-------
niter : iterator
    An iterator over nodes in nbunch that are also in the graph.
    If nbunch is None, iterate over all nodes in the graph.

Raises
------
NetworkXError
    If nbunch is not a node or sequence of nodes.
    If a node in nbunch is not hashable.

See Also
--------
Graph.__iter__

Notes
-----
When nbunch is an iterator, the returned iterator yields values
directly from nbunch, becoming exhausted when nbunch is exhausted.

To test whether nbunch is a single node, one can use
"if nbunch in self:", even after processing with this routine.

If nbunch is not a node or a (possibly empty) sequence/iterator
or None, a :exc:`NetworkXError` is raised.  Also, if any object in
nbunch is not hashable, a :exc:`NetworkXError` is raised.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def nbunch_iter(self, nbunch=None):
	    
	    if nbunch is None:  # include all nodes via iterator
	        bunch = iter(self._adj)
	    elif nbunch in self:  # if nbunch is a single node
	        bunch = iter([nbunch])
	    else:  # if nbunch is a sequence of nodes
	
	        def bunch_iter(nlist, adj):
	            try:
	                for n in nlist:
	                    if n in adj:
	                        yield n
	            except TypeError as e:
	                exc, message = e, e.args[0]
	                # capture error for non-sequence/iterator nbunch.
	                if "iter" in message:
	                    exc = NetworkXError(
	                        "nbunch is not a node or a sequence of nodes."
	                    )
	                # capture error for unhashable node.
	                if "hashable" in message:
	                    exc = NetworkXError(
	                        f"Node {n} in sequence nbunch is not a valid node."
	                    )
	                raise exc
	
	        bunch = bunch_iter(nbunch, self._adj)
	    return bunch
	
	```

______

### *BipartiteGraph*.**neighbors**`#!py3 (self, n)` { #neighbors data-toc-label=neighbors }

Returns an iterator over all neighbors of node n.

This is identical to `iter(G[n])`

Parameters
----------
n : node
   A node in the graph

Returns
-------
neighbors : iterator
    An iterator over all neighbors of node n

Raises
------
NetworkXError
    If the node n is not in the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> [n for n in G.neighbors(0)]
[1]

Notes
-----
Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:

>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_edge("a", "b", weight=7)
>>> G["a"]
AtlasView({'b': {'weight': 7}})
>>> G = nx.path_graph(4)
>>> [n for n in G[0]]
[1]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def neighbors(self, n):
	    
	    try:
	        return iter(self._adj[n])
	    except KeyError as e:
	        raise NetworkXError(f"The node {n} is not in the graph.") from e
	
	```

______

### *BipartiteGraph*.**number_of_edges**`#!py3 (self, u=None, v=None)` { #number_of_edges data-toc-label=number_of_edges }

Returns the number of edges between two nodes.

Parameters
----------
u, v : nodes, optional (default=all edges)
    If u and v are specified, return the number of edges between
    u and v. Otherwise return the total number of all edges.

Returns
-------
nedges : int
    The number of edges in the graph.  If nodes `u` and `v` are
    specified return the number of edges between those nodes. If
    the graph is directed, this only returns the number of edges
    from `u` to `v`.

See Also
--------
size

Examples
--------
For undirected graphs, this method counts the total number of
edges in the graph:

>>> G = nx.path_graph(4)
>>> G.number_of_edges()
3

If you specify two nodes, this counts the total number of edges
joining the two nodes:

>>> G.number_of_edges(0, 1)
1

For directed graphs, this method can count the total number of
directed edges from `u` to `v`:

>>> G = nx.DiGraph()
>>> G.add_edge(0, 1)
>>> G.add_edge(1, 0)
>>> G.number_of_edges(0, 1)
1


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def number_of_edges(self, u=None, v=None):
	    
	    if u is None:
	        return int(self.size())
	    if v in self._adj[u]:
	        return 1
	    return 0
	
	```

______

### *BipartiteGraph*.**number_of_nodes**`#!py3 (self)` { #number_of_nodes data-toc-label=number_of_nodes }

Returns the number of nodes in the graph.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
order: identical method
__len__: identical method

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.number_of_nodes()
3


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def number_of_nodes(self):
	    
	    return len(self._node)
	
	```

______

### *BipartiteGraph*.**order**`#!py3 (self)` { #order data-toc-label=order }

Returns the number of nodes in the graph.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
number_of_nodes: identical method
__len__: identical method

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.order()
3


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def order(self):
	    
	    return len(self._node)
	
	```

______

### *BipartiteGraph*.**print_network_statistics**`#!py3 (self, title)` { #print_network_statistics data-toc-label=print_network_statistics }

Prints that statistics of the bipartite graph (e.g., number of actor/community vertices). This is useful for
sanity checking that the graph has been populated correctly.

**Parameters**
> **title:** ``str`` -- The title of the table.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def print_network_statistics(self, title):
	    
	
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
	
	```

______

### *BipartiteGraph*.**remove_edge**`#!py3 (self, u, v)` { #remove_edge data-toc-label=remove_edge }

Remove the edge between u and v.

Parameters
----------
u, v : nodes
    Remove the edge between nodes u and v.

Raises
------
NetworkXError
    If there is not an edge between u and v.

See Also
--------
remove_edges_from : remove a collection of edges

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, etc
>>> G.remove_edge(0, 1)
>>> e = (1, 2)
>>> G.remove_edge(*e)  # unpacks e from an edge tuple
>>> e = (2, 3, {"weight": 7})  # an edge with attribute data
>>> G.remove_edge(*e[:2])  # select first part of edge tuple


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def remove_edge(self, u, v):
	    
	    try:
	        del self._adj[u][v]
	        if u != v:  # self-loop needs only one entry removed
	            del self._adj[v][u]
	    except KeyError as e:
	        raise NetworkXError(f"The edge {u}-{v} is not in the graph") from e
	
	```

______

### *BipartiteGraph*.**remove_edges_from**`#!py3 (self, ebunch)` { #remove_edges_from data-toc-label=remove_edges_from }

Remove all edges specified in ebunch.

Parameters
----------
ebunch: list or container of edge tuples
    Each edge given in the list or container will be removed
    from the graph. The edges can be:

        - 2-tuples (u, v) edge between u and v.
        - 3-tuples (u, v, k) where k is ignored.

See Also
--------
remove_edge : remove a single edge

Notes
-----
Will fail silently if an edge in ebunch is not in the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> ebunch = [(1, 2), (2, 3)]
>>> G.remove_edges_from(ebunch)


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def remove_edges_from(self, ebunch):
	    
	    adj = self._adj
	    for e in ebunch:
	        u, v = e[:2]  # ignore edge data if present
	        if u in adj and v in adj[u]:
	            del adj[u][v]
	            if u != v:  # self loop needs only one entry removed
	                del adj[v][u]
	
	```

______

### *BipartiteGraph*.**remove_node**`#!py3 (self, n)` { #remove_node data-toc-label=remove_node }

Remove node n.

Removes the node n and all adjacent edges.
Attempting to remove a non-existent node will raise an exception.

Parameters
----------
n : node
   A node in the graph

Raises
------
NetworkXError
   If n is not in the graph.

See Also
--------
remove_nodes_from

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> list(G.edges)
[(0, 1), (1, 2)]
>>> G.remove_node(1)
>>> list(G.edges)
[]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def remove_node(self, n):
	    
	    adj = self._adj
	    try:
	        nbrs = list(adj[n])  # list handles self-loops (allows mutation)
	        del self._node[n]
	    except KeyError as e:  # NetworkXError if n not in self
	        raise NetworkXError(f"The node {n} is not in the graph.") from e
	    for u in nbrs:
	        del adj[u][n]  # remove all edges n-u in graph
	    del adj[n]  # now remove node
	
	```

______

### *BipartiteGraph*.**remove_nodes_from**`#!py3 (self, nodes)` { #remove_nodes_from data-toc-label=remove_nodes_from }

Remove multiple nodes.

Parameters
----------
nodes : iterable container
    A container of nodes (list, dict, set, etc.).  If a node
    in the container is not in the graph it is silently
    ignored.

See Also
--------
remove_node

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> e = list(G.nodes)
>>> e
[0, 1, 2]
>>> G.remove_nodes_from(e)
>>> list(G.nodes)
[]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def remove_nodes_from(self, nodes):
	    
	    adj = self._adj
	    for n in nodes:
	        try:
	            del self._node[n]
	            for u in list(adj[n]):  # list handles self-loops
	                del adj[u][n]  # (allows mutation of dict in loop)
	            del adj[n]
	        except KeyError:
	            pass
	
	```

______

### *BipartiteGraph*.**set_vertex_colour**`#!py3 (self, colour_dict, actor=True)` { #set_vertex_colour data-toc-label=set_vertex_colour }

Sets what colour the vertex should be represented within the BGE Plot.

**Parameters**
> **colour_dict:** ``dict`` --  A dictionary containing a selected colour for each vertex in either the actor
or community set; where key=actor label, value=colour.

> E.g., colour_dict = {'a1': 'G', 'a2: 'B'}

> **actor:** ``bool`` -- Whether to set the colour for the actor or community set.

!!! note
    A colour for all vertices within the actor or community set must be selected.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def set_vertex_colour(self, colour_dict, actor=True):
	    
	    # Selected either the actor or community vertices.
	    key, vertices = ('a', self.actors) if actor else ('c', self.comms)
	
	    self.colours[key] = [colour_dict[vertex] for vertex in vertices]
	
	```

______

### *BipartiteGraph*.**size**`#!py3 (self, weight=None)` { #size data-toc-label=size }

Returns the number of edges or total of all edge weights.

Parameters
----------
weight : string or None, optional (default=None)
    The edge attribute that holds the numerical value used
    as a weight. If None, then each edge has weight 1.

Returns
-------
size : numeric
    The number of edges or
    (if weight keyword is provided) the total weight sum.

    If weight is None, returns an int. Otherwise a float
    (or more general numeric if the weights are more general).

See Also
--------
number_of_edges

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.size()
3

>>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_edge("a", "b", weight=2)
>>> G.add_edge("b", "c", weight=4)
>>> G.size()
2
>>> G.size(weight="weight")
6.0


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def size(self, weight=None):
	    
	    s = sum(d for v, d in self.degree(weight=weight))
	    # If `weight` is None, the sum of the degrees is guaranteed to be
	    # even, so we can perform integer division and hence return an
	    # integer. Otherwise, the sum of the weighted degrees is not
	    # guaranteed to be an integer, so we perform "real" division.
	    return s // 2 if weight is None else s / 2
	
	```

______

### *BipartiteGraph*.**subgraph**`#!py3 (self, nodes)` { #subgraph data-toc-label=subgraph }

Returns a SubGraph view of the subgraph induced on `nodes`.

The induced subgraph of the graph contains the nodes in `nodes`
and the edges between those nodes.

Parameters
----------
nodes : list, iterable
    A container of nodes which will be iterated through once.

Returns
-------
G : SubGraph View
    A subgraph view of the graph. The graph structure cannot be
    changed but node/edge attributes can and are shared with the
    original graph.

Notes
-----
The graph, edge and node attributes are shared with the original graph.
Changes to the graph structure is ruled out by the view, but changes
to attributes are reflected in the original graph.

To create a subgraph with its own copy of the edge/node attributes use:
G.subgraph(nodes).copy()

For an inplace reduction of a graph to a subgraph you can remove nodes:
G.remove_nodes_from([n for n in G if n not in set(nodes)])

Subgraph views are sometimes NOT what you want. In most cases where
you want to do more than simply look at the induced edges, it makes
more sense to just create the subgraph as its own graph with code like:

::

    # Create a subgraph SG based on a (possibly multigraph) G
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, keydict in nbrs.items() if nbr in largest_wcc
            for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> H = G.subgraph([0, 1, 2])
>>> list(H.edges)
[(0, 1), (1, 2)]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def subgraph(self, nodes):
	    
	    induced_nodes = nx.filters.show_nodes(self.nbunch_iter(nodes))
	    # if already a subgraph, don't make a chain
	    subgraph = nx.graphviews.subgraph_view
	    if hasattr(self, "_NODE_OK"):
	        return subgraph(self._graph, induced_nodes, self._EDGE_OK)
	    return subgraph(self, induced_nodes)
	
	```

______

### *BipartiteGraph*.**to_directed**`#!py3 (self, as_view=False)` { #to_directed data-toc-label=to_directed }

Returns a directed representation of the graph.

Returns
-------
G : DiGraph
    A directed graph with the same name, same nodes, and with
    each edge (u, v, data) replaced by two directed edges
    (u, v, data) and (v, u, data).

Notes
-----
This returns a "deepcopy" of the edge, node, and
graph attributes which attempts to completely copy
all of the data and references.

This is in contrast to the similar D=DiGraph(G) which returns a
shallow copy of the data.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/3/library/copy.html.

Warning: If you have subclassed Graph to use dict-like objects
in the data structure, those changes do not transfer to the
DiGraph created by this method.

Examples
--------
>>> G = nx.Graph()  # or MultiGraph, etc
>>> G.add_edge(0, 1)
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1), (1, 0)]

If already directed, return a (deep) copy

>>> G = nx.DiGraph()  # or MultiDiGraph, etc
>>> G.add_edge(0, 1)
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1)]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def to_directed(self, as_view=False):
	    
	    graph_class = self.to_directed_class()
	    if as_view is True:
	        return nx.graphviews.generic_graph_view(self, graph_class)
	    # deepcopy when not a view
	    G = graph_class()
	    G.graph.update(deepcopy(self.graph))
	    G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
	    G.add_edges_from(
	        (u, v, deepcopy(data))
	        for u, nbrs in self._adj.items()
	        for v, data in nbrs.items()
	    )
	    return G
	
	```

______

### *BipartiteGraph*.**to_directed_class**`#!py3 (self)` { #to_directed_class data-toc-label=to_directed_class }

Returns the class to use for empty directed copies.

If you subclass the base classes, use this to designate
what directed class to use for `to_directed()` copies.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def to_directed_class(self):
	    
	    return nx.DiGraph
	
	```

______

### *BipartiteGraph*.**to_undirected**`#!py3 (self, as_view=False)` { #to_undirected data-toc-label=to_undirected }

Returns an undirected copy of the graph.

Parameters
----------
as_view : bool (optional, default=False)
  If True return a view of the original undirected graph.

Returns
-------
G : Graph/MultiGraph
    A deepcopy of the graph.

See Also
--------
Graph, copy, add_edge, add_edges_from

Notes
-----
This returns a "deepcopy" of the edge, node, and
graph attributes which attempts to completely copy
all of the data and references.

This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
shallow copy of the data.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/3/library/copy.html.

Warning: If you have subclassed DiGraph to use dict-like objects
in the data structure, those changes do not transfer to the
Graph created by this method.

Examples
--------
>>> G = nx.path_graph(2)  # or MultiGraph, etc
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1), (1, 0)]
>>> G2 = H.to_undirected()
>>> list(G2.edges)
[(0, 1)]


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def to_undirected(self, as_view=False):
	    
	    graph_class = self.to_undirected_class()
	    if as_view is True:
	        return nx.graphviews.generic_graph_view(self, graph_class)
	    # deepcopy when not a view
	    G = graph_class()
	    G.graph.update(deepcopy(self.graph))
	    G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
	    G.add_edges_from(
	        (u, v, deepcopy(d))
	        for u, nbrs in self._adj.items()
	        for v, d in nbrs.items()
	    )
	    return G
	
	```

______

### *BipartiteGraph*.**to_undirected_class**`#!py3 (self)` { #to_undirected_class data-toc-label=to_undirected_class }

Returns the class to use for empty undirected copies.

If you subclass the base classes, use this to designate
what directed class to use for `to_directed()` copies.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def to_undirected_class(self):
	    
	    return Graph
	
	```

______

### *BipartiteGraph*.**update**`#!py3 (self, edges=None, nodes=None)` { #update data-toc-label=update }

Update the graph using nodes/edges/graphs as input.

Like dict.update, this method takes a graph as input, adding the
graph's nodes and edges to this graph. It can also take two inputs:
edges and nodes. Finally it can take either edges or nodes.
To specify only nodes the keyword `nodes` must be used.

The collections of edges and nodes are treated similarly to
the add_edges_from/add_nodes_from methods. When iterated, they
should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).

Parameters
----------
edges : Graph object, collection of edges, or None
    The first parameter can be a graph or some edges. If it has
    attributes `nodes` and `edges`, then it is taken to be a
    Graph-like object and those attributes are used as collections
    of nodes and edges to be added to the graph.
    If the first parameter does not have those attributes, it is
    treated as a collection of edges and added to the graph.
    If the first argument is None, no edges are added.
nodes : collection of nodes, or None
    The second parameter is treated as a collection of nodes
    to be added to the graph unless it is None.
    If `edges is None` and `nodes is None` an exception is raised.
    If the first parameter is a Graph, then `nodes` is ignored.

Examples
--------
>>> G = nx.path_graph(5)
>>> G.update(nx.complete_graph(range(4, 10)))
>>> from itertools import combinations
>>> edges = (
...     (u, v, {"power": u * v})
...     for u, v in combinations(range(10, 20), 2)
...     if u * v < 225
... )
>>> nodes = [1000]  # for singleton, use a container
>>> G.update(edges, nodes)

Notes
-----
It you want to update the graph using an adjacency structure
it is straightforward to obtain the edges/nodes from adjacency.
The following examples provide common cases, your adjacency may
be slightly different and require tweaks of these examples::

>>> # dict-of-set/list/tuple
>>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
>>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
>>> G.update(edges=e, nodes=adj)

>>> DG = nx.DiGraph()
>>> # dict-of-dict-of-attribute
>>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
>>> e = [
...     (u, v, {"weight": d})
...     for u, nbrs in adj.items()
...     for v, d in nbrs.items()
... ]
>>> DG.update(edges=e, nodes=adj)

>>> # dict-of-dict-of-dict
>>> adj = {1: {2: {"weight": 1.3}, 3: {"color": 0.7, "weight": 1.2}}}
>>> e = [
...     (u, v, {"weight": d})
...     for u, nbrs in adj.items()
...     for v, d in nbrs.items()
... ]
>>> DG.update(edges=e, nodes=adj)

>>> # predecessor adjacency (dict-of-set)
>>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
>>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]

>>> # MultiGraph dict-of-dict-of-dict-of-attribute
>>> MDG = nx.MultiDiGraph()
>>> adj = {
...     1: {2: {0: {"weight": 1.3}, 1: {"weight": 1.2}}},
...     3: {2: {0: {"weight": 0.7}}},
... }
>>> e = [
...     (u, v, ekey, d)
...     for u, nbrs in adj.items()
...     for v, keydict in nbrs.items()
...     for ekey, d in keydict.items()
... ]
>>> MDG.update(edges=e)

See Also
--------
add_edges_from: add multiple edges to a graph
add_nodes_from: add multiple nodes to a graph


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def update(self, edges=None, nodes=None):
	    
	    if edges is not None:
	        if nodes is not None:
	            self.add_nodes_from(nodes)
	            self.add_edges_from(edges)
	        else:
	            # check if edges is a Graph object
	            try:
	                graph_nodes = edges.nodes
	                graph_edges = edges.edges
	            except AttributeError:
	                # edge not Graph-like
	                self.add_edges_from(edges)
	            else:  # edges is Graph-like
	                self.add_nodes_from(graph_nodes.data())
	                self.add_edges_from(graph_edges.data())
	                self.graph.update(edges.graph)
	    elif nodes is not None:
	        self.add_nodes_from(nodes)
	    else:
	        raise NetworkXError("update needs nodes or edges input")
	
	```

______


______

