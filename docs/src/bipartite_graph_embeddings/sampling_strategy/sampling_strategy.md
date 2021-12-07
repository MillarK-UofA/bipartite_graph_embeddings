## **SamplingStrategy**`#!py3 class` { #SamplingStrategy data-toc-label=SamplingStrategy }

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

### *SamplingStrategy*.**__init__**`#!py3 (self, batch_size, ns, large_dataset=True)` { #__init__ data-toc-label=__init__ }

Defines the parent edge sampling class.

**Parameters**
> **batch_size:** ``int`` -- The number of positive edges to evaluate at once.

> **ns:** ``int`` -- The number of negative samples to generate per positive sample.

> **large_dataset:** ``bool`` -- Whether to store the generated dataset in memory (large_dataset=False) or on
disk (large_dataset=True).


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, batch_size, ns, large_dataset=True):
	    
	
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
	
	```

______

### *SamplingStrategy*.**__len__**`#!py3 (self)` { #__len__ data-toc-label=__len__ }

Defines the length of the SamplingStrategy by the length of the generated dataset.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __len__(self):
	    
	    return len(self.dataset)
	
	```

______

### *SamplingStrategy*.**allocate_dataset**`#!py3 (self, shape)` { #allocate_dataset data-toc-label=allocate_dataset }

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

### *SamplingStrategy*.**generate_corpus**`#!py3 (self, graph)` { #generate_corpus data-toc-label=generate_corpus }

Evaluates the bipartite graph structure to construct a dataset to train the BGE model. generate_corpus is
overridden by all child classes.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_corpus(self, graph):
	    
	    pass
	
	```

______

### *SamplingStrategy*.**get_batch**`#!py3 (self, idx)` { #get_batch data-toc-label=get_batch }

Gets the next batch of positive and negative samples. get_batch is overridden by all child classes.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_batch(self, idx):
	    
	    pass
	
	```

______

### *SamplingStrategy*.**get_edge_weight**`#!py3 (graph, v, u)` { #get_edge_weight data-toc-label=get_edge_weight }

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

### *SamplingStrategy*.**shuffle**`#!py3 (self)` { #shuffle data-toc-label=shuffle }

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

