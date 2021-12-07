
The random walk sampling strategy is heavily based on node2vec.

The code for node2vec has been sourced from GroverA et al. The github repository for node2vec can be
found: https://github.com/aditya-grover/node2vec.

@inproceedings{node2vec-kdd2016,
    author = {Grover, Aditya and Leskovec, Jure},
    title = {node2vec: Scalable Feature Learning for Networks},
    booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    year = {2016}
}

______

## **RandomWalk**`#!py3 class` { #RandomWalk data-toc-label=RandomWalk }

Defines the parent edge sampling strategy. Contains common functions which inherited by all edge sampling
strategies.

**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`__len__`](#__len__)
 - [`allocate_dataset`](#allocate_dataset)
 - [`generate_corpus`](#generate_corpus)
 - [`get_alias_edge`](#get_alias_edge)
 - [`get_batch`](#get_batch)
 - [`get_edge_weight`](#get_edge_weight)
 - [`node2vec_walk`](#node2vec_walk)
 - [`preprocess_transition_probs`](#preprocess_transition_probs)
 - [`shuffle`](#shuffle)

### *RandomWalk*.**__init__**`#!py3 (self, graph, batch_size, ns)` { #__init__ data-toc-label=__init__ }

Creates a dataloader object from a given bipartite graph


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, graph, batch_size, ns):
	    
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
	    self.save_path = os.path.join(self.temp_dataset_path, str(graph) + "_rw.dat")
	
	    # Generate the dataset (positive and negative samples).
	    self.generate_corpus(graph)
	
	```

______

### *RandomWalk*.**__len__**`#!py3 (self)` { #__len__ data-toc-label=__len__ }

Defines the length of the SamplingStrategy by the length of the generated dataset.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __len__(self):
	    
	    return len(self.dataset)
	
	```

______

### *RandomWalk*.**allocate_dataset**`#!py3 (self, shape)` { #allocate_dataset data-toc-label=allocate_dataset }

Evaluates the bipartite graph structure to construct a dataset to train the BGE model. generate_corpus is
overridden by all child classes.

!!! Note
    The use of np.uintc as the element type for the dataset variable does restrict the number of nodes that can
    exist in the actor and community sets. However, in its current state, there can be up to 4.3B actors and
    4.3B communities.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def allocate_dataset(self, shape):
	    
	
	    # - Allocate memory to store the dataset --------------------------------------------------------------------- #
	    # If large_false=true, store dataset on disk; else, store dataset in memory.
	    if not self.large_dataset:
	        self.dataset = np.zeros(shape=shape, dtype=np.uintc)
	    else:
	        self.dataset = np.memmap(self.save_path, dtype=np.uintc, mode='w+', shape=shape)
	
	```

______

### *RandomWalk*.**generate_corpus**`#!py3 (self, graph)` { #generate_corpus data-toc-label=generate_corpus }

Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
BGE algorithm.

**Returns**
> **edges:** ``array`` -- A 2D array representing the edges of the graph.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_corpus(self, graph):
	    
	
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
	
	    # Initialise progress bar.
	    bar = tqdm(total=self.sampling_budget)
	
	    nodes = list(graph.nodes())
	    idx = 0
	    for walk_iter in range(self.r):
	
	        # Shuffle nodes.
	        random.shuffle(nodes)
	
	        for node in nodes:
	
	            # Compute walk
	            walk = self.node2vec_walk(graph, node, alias_nodes, alias_edges)
	
	            # If the walk starts with an actor or community.
	            offset = 0 if node in graph.actor_idx else 1
	
	            # For each step in the walk.
	            for step_idx in range(self.l-1):
	
	                # draw positive edge (actor, community).
	                self.dataset[idx, 0] = graph.actor_idx[walk[step_idx + (step_idx + offset) % 2]]
	                self.dataset[idx, 1] = graph.comm_idx[walk[step_idx + (1 - (step_idx + offset) % 2)]]
	
	                # draw negative community examples.
	                for i in range(self.ns):
	                    self.dataset[idx, 2 + i] = alias_draw(J_neg, q_neg)
	
	                idx += 1
	                bar.update()
	    # ------------------------------------------------------------------------------------------------------------ #
	
	    # Store dataset on GPU.
	    if not self.large_dataset:
	        self.dataset = xp.array(self.dataset)
	
	    print("sampling time: {:,} seconds".format(int(time() - start_time)))
	
	```

______

### *RandomWalk*.**get_alias_edge**`#!py3 (self, graph, src, dst)` { #get_alias_edge data-toc-label=get_alias_edge }

Get the alias edge setup lists for a given edge.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_alias_edge(self, graph, src, dst):
	    
	
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
	
	```

______

### *RandomWalk*.**get_batch**`#!py3 (self, idx)` { #get_batch data-toc-label=get_batch }

Gets the next batch.

**Parameters**
> **idx:** ``int`` -- the starting index of the current batch.

**Returns**
> **actor_batch:** ``np.array`` or ``cp.array`` -- the current actor vertices in the current batch.

> **comm_batch:** ``np.array`` or ``cp.array`` -- the current community vertices in the current batch.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_batch(self, idx):
	    
	
	    batch = xp.array(self.dataset[idx:idx+self.batch_size, :])
	
	    return batch[:, :1], batch[:, 1:]
	
	```

______

### *RandomWalk*.**get_edge_weight**`#!py3 (graph, v, u)` { #get_edge_weight data-toc-label=get_edge_weight }

Returns the edge weight between two vertices. Returns 1 if the graph is unweighted.

**Parameters**
> **graph:** ``common.BipartiteGraph`` -- The graph under analysis.

> **v:** ``str`` -- The first vertex.

> **u:** ``str`` -- The second vertex.

**Returns**
> **weight:** ``int`` -- The weight of the edge. Weight is set to 1 if the graph is unweighted.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@staticmethod
	def get_edge_weight(graph, v, u):
	    
	    if graph.weighted:
	        return graph[v][u]['weight']
	    else:
	        return 1
	
	```

______

### *RandomWalk*.**node2vec_walk**`#!py3 (self, graph, start_node, alias_nodes, alias_edges)` { #node2vec_walk data-toc-label=node2vec_walk }

Simulate a random walk starting from particular starting node.

**Parameters**
> **graph:** ``common.BipartiteGraph`` -- The graph under analysis.

> **start_node:** ``int`` -- The index of the vertex to start the walk.

> **walk:** ``list`` -- The list of vertices that comprises the walk.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def node2vec_walk(self, graph, start_node, alias_nodes, alias_edges):
	    
	
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
	
	```

______

### *RandomWalk*.**preprocess_transition_probs**`#!py3 (self, graph)` { #preprocess_transition_probs data-toc-label=preprocess_transition_probs }

Preprocessing of transition probabilities for guiding the random walks.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def preprocess_transition_probs(self, graph):
	    
	
	    alias_nodes = {}
	    for node in graph.nodes():
	        unnormalized_probs = [graph[node][nbr].get('weight', 1) for nbr in sorted(graph.neighbors(node))]
	        norm_const = sum(unnormalized_probs)
	        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
	        alias_nodes[node] = alias_setup(normalized_probs)
	    alias_edges = {}
	
	    # Undirected graph. Add an alias edge for both directions.
	    for edge in graph.edges():
	        alias_edges[edge] = self.get_alias_edge(graph, edge[0], edge[1])
	        alias_edges[(edge[1], edge[0])] = self.get_alias_edge(graph, edge[1], edge[0])
	
	    return alias_nodes, alias_edges
	
	```

______

### *RandomWalk*.**shuffle**`#!py3 (self)` { #shuffle data-toc-label=shuffle }

shuffles the dataset.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def shuffle(self):
	    
	    # Note: Cupy is much faster at shuffling the edge list than numpy.
	    perm = self.operator.random.permutation(len(self.dataset))
	
	    # Shuffle edges.
	    self.dataset = self.dataset[perm]
	
	```

______


______

