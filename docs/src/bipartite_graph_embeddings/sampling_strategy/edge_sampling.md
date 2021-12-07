## **EdgeSampler**`#!py3 class` { #EdgeSampler data-toc-label=EdgeSampler }

Defines the parent edge sampling strategy. Contains common functions which inherited by all edge sampling
strategies.

**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`__len__`](#__len__)
 - [`allocate_dataset`](#allocate_dataset)
 - [`generate_corpus`](#generate_corpus)
 - [`get_batch`](#get_batch)
 - [`get_edge_weight`](#get_edge_weight)
 - [`shuffle`](#shuffle)

### *EdgeSampler*.**__init__**`#!py3 (self, graph, batch_size, ns)` { #__init__ data-toc-label=__init__ }

Defines the conventional edge sampling strategy.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, graph, batch_size, ns):
	    
	    super().__init__(batch_size, ns)
	
	    # The number of edges to evaluate in an epoch.
	    self.sampling_budget = 10 * 80 * len(graph)
	
	    # The temporary storage of the dataset (if using large_dataset=True)
	    self.save_path = os.path.join(self.temp_dataset_path, str(graph)+"_es.dat")
	
	    # Generate the dataset (positive and negative samples).
	    self.generate_corpus(graph)
	
	```

______

### *EdgeSampler*.**__len__**`#!py3 (self)` { #__len__ data-toc-label=__len__ }

Defines the length of the SamplingStrategy by the length of the generated dataset.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __len__(self):
	    
	    return len(self.dataset)
	
	```

______

### *EdgeSampler*.**allocate_dataset**`#!py3 (self, shape)` { #allocate_dataset data-toc-label=allocate_dataset }

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

### *EdgeSampler*.**generate_corpus**`#!py3 (self, graph)` { #generate_corpus data-toc-label=generate_corpus }

Generates the edgelist of the Bipartite Graph using the vertices' indices. This is used as the corpus for
BGE algorithm.

**Parameters**
> **graph:** ``common.BipartiteGraph`` -- The graph under analysis.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_corpus(self, graph):
	    
	
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
	
	```

______

### *EdgeSampler*.**get_batch**`#!py3 (self, idx)` { #get_batch data-toc-label=get_batch }

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

### *EdgeSampler*.**get_edge_weight**`#!py3 (graph, v, u)` { #get_edge_weight data-toc-label=get_edge_weight }

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

### *EdgeSampler*.**shuffle**`#!py3 (self)` { #shuffle data-toc-label=shuffle }

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

