
Defines and trains the bipartite graph embedding (BGE) model.

______

## **BGE**`#!py3 class` { #BGE data-toc-label=BGE }

Bipartite Graph Embeddings (BGE)

**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`get_embeddings`](#get_embeddings)
 - [`load_embeddings_from_file`](#load_embeddings_from_file)
 - [`save_embeddings`](#save_embeddings)
 - [`train_model`](#train_model)

### *BGE*.**__init__**`#!py3 (self, graph, embedding_dim=128, lf='dot', opt='sgd', alpha=0.025, init='normal', update_actors=True, update_comms=True)` { #__init__ data-toc-label=__init__ }

Initialises the BGE class.

**Parameters**
> **graph:** ``BipartiteGraph`` -- The graph to train BGE on.

> **embedding_dim:** ``int`` -- The number of dimensions of each embedding.

> **lf:** ``string`` -- The loss function to use. Either: "cosine" or "dot". Default "cosine".

> **opt:** ``string`` -- The optimisation function to use. Either "adam" or "sgd". Default "adam".

> **alpha:** ``float`` -- The learning rate. Default 0.025.

> **init:** ``string`` -- Which method to use to initialise graph embeddings.

> **update_actors:** ``bool`` -- Whether to update the actor embeddings.

> **update_comms:** ``bool`` -- Whether to update the community embeddings.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, graph, embedding_dim=128, lf='dot', opt='sgd', alpha=0.025, init='normal',
	             update_actors=True, update_comms=True):
	    
	
	    self.graph = graph
	
	    # get the number of actors and communities in the graph.
	    self.num_actors = len(self.graph.actors)
	    self.num_comms = len(self.graph.comms)
	
	    # Set the dimensionality of the embedding space.
	    self.embedding_dim = embedding_dim
	
	    # Defines how the starting weights (embeddings) are initialised.
	    initialiser = Initialiser(self.num_actors, self.num_comms, embedding_dim)
	    self.W = initialiser.populate(init)
	
	    # Defines which loss function will be used.
	    self.loss_function = self.loss_functions[lf]
	
	    # Defines which optimiser will be used.
	    self.optimiser = self.optimisers[opt](alpha, self.num_actors, self.num_comms, embedding_dim)
	
	    # Whether to update the actor/community embeddings.
	    self.update_actors = update_actors
	    self.update_comms = update_comms
	
	```

______

### *BGE*.**get_embeddings**`#!py3 (self)` { #get_embeddings data-toc-label=get_embeddings }

Extract actor/community embeddings. Save to a pandas dataframe. A pandas dataframe is used as it is quicker to
save to a csv file.

**Returns**
> **df_actors:** ``pandas.DataFrame`` -- dataframe containing the embeddings of each actor.

> **df_comms:** ``pandas.DataFrame`` -- dataframe containing the embeddings of each community.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def get_embeddings(self):
	    
	
	    # If using cupy, convert embeddings to numpy arrays before plotting them. While pandas words with cupy arrays,
	    # it is much slower.
	    w_a = self.W['a'] if not _cupy_available else xp.asnumpy(xp_round(self.W['a']))
	    w_c = self.W['c'] if not _cupy_available else xp.asnumpy(xp_round(self.W['c']))
	
	    df_actors = pd.DataFrame(data=w_a, index=self.graph.actors)
	    df_comms = pd.DataFrame(data=w_c, index=self.graph.comms)
	
	    return df_actors, df_comms
	
	```

______

### *BGE*.**load_embeddings_from_file**`#!py3 (self, file_path, verbose=False)` { #load_embeddings_from_file data-toc-label=load_embeddings_from_file }

Loads pre-computed embeddings from a specified file path. Can be used to continue training. Vertices in the
graph that are not in the embedding file are initialised randomly.

**Parameters**
> **file_path:** ``str`` -- The path to the desired embeddings file.

> **verbose:** ``bool`` -- Whether to print overall statistics about the found  embeddings.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def load_embeddings_from_file(self, file_path, verbose=False):
	    
	
	    # - Locate/Open JSON file form specified path. --------------------------------------------------------------- #
	    try:
	        with open(file_path, 'r', encoding='utf8') as json_file:
	            embedding_dict = json.load(json_file)
	    except FileNotFoundError:
	        print("Embedding file could not be located at: {}".format(file_path))
	        exit()
	    except json.decoder.JSONDecodeError:
	        print("Specified file is incorrectly formatted.")
	        exit()
	
	    # Check that the dimension of the saved embeddings matches the desired dimensions. Checks the first embedding.
	    saved_embedding_dim = len(embedding_dict[next(iter(embedding_dict))])
	    if saved_embedding_dim != self.embedding_dim:
	        print("Specified embedding dimension ({:,}) does not match the dimension of saved embeddings ({:,})."
	              .format(self.embedding_dim, saved_embedding_dim))
	        exit()
	    # ------------------------------------------------------------------------------------------------------------ #
	
	    # - Restore embeddings from JSON file. ----------------------------------------------------------------------- #
	    print("Restoring Embeddings from: {}...".format(file_path))
	
	    vertex_dict = {'a': self.graph.actors, 'c': self.graph.comms}
	
	    # For both the actor and community sets, check for known embeddings.
	    for vertex_type, vertices in vertex_dict.items():
	
	        restore_count = 0
	        for idx, vertex in enumerate(vertices):
	
	            # Check if embedding is known (i.e., in the embedding_dict).
	            saved_embedding = embedding_dict.get(vertex, None)
	
	            # If embedding is known.
	            if saved_embedding is not None:
	                self.W[vertex_type][idx] = xp.array(saved_embedding)
	                restore_count += 1
	                continue
	
	            # If embedding is unknown, the embedding is randomly initialised.
	            self.W[vertex_type][idx] = xp.random.normal(loc=0, scale=0.01, size=(1, self.embedding_dim))
	
	        if verbose:
	            vertex_label = "actor" if vertex_type == 'a' else "community"
	            percentage = restore_count / len(vertices) * 100
	            print("\t - Known embeddings in {} set: {:,} ({:.2f} %)".format(vertex_label, restore_count, percentage))
	
	```

______

### *BGE*.**save_embeddings**`#!py3 (self, save_path, verbose=True)` { #save_embeddings data-toc-label=save_embeddings }

Saves the trained embeddings to a csv file.

**Parameters**
> **save_path:** ``string`` -- desired save path to the csv file.

> **verbose:** ``bool`` -- Whether to print overall statistics about the generated embeddings.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def save_embeddings(self, save_path, verbose=True):
	    
	
	    df_actors, df_comms = self.get_embeddings()
	
	    if verbose:
	        print("Actor Embeddings: max: {}, min: {}".format(df_actors.max().max(), df_actors.min().min()))
	        print("Comm Embeddings: max: {}, min: {}".format(df_comms.max().max(), df_comms.min().min()))
	
	    # - Save Embeddings to JSON file. ---------------------------------------------------------------------------- #
	    df = pd.concat([df_actors, df_comms])
	
	    # transform data frame to dictionary.
	    weights = df.to_dict(orient="split")
	
	    # format dictionary into desired format for the json file.
	    weight_json = {vertex_id: embedding for vertex_id, embedding in zip(weights['index'], weights['data'])}
	
	    # save the json file.
	    with open(save_path, 'w') as save_file:
	        json.dump(weight_json, save_file)
	
	```

______

### *BGE*.**train_model**`#!py3 (self, batch_size, ns, sampling, epoch_sample_size, max_epochs)` { #train_model data-toc-label=train_model }

Trains the graph embeddings.

**Parameters**
> **batch_size:** ``int`` -- The number of forward/backward passes to do at once.

> **ns:** ``int`` -- The number of negative samples to use in each forward/backward pass.

> **sampling:** ``str`` -- The sampling strategy to use. Either 'tes' - transient edge sampling, 'es' -
 edge sampling, 'rw' - random walks.)

> **epoch_sample_size:** ``int`` -- The number of epochs to run before checking if the stop condition has been
met. Averages results of all epochs in the sample size.

> **max_epochs:** ``int`` or ``None`` -- The number of epochs to run. If ``None``, training is will continue
until the stop condition is met.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def train_model(self, batch_size, ns, sampling, epoch_sample_size, max_epochs):
	    
	
	    # Define the edge sampling strategy to use.
	    dataloader = self.sampling_strategies[sampling](self.graph, batch_size, ns)
	
	    # Set the stop condition for the current training.
	    stop_condition = StopCondition(epoch_sample_size=epoch_sample_size, max_epochs=max_epochs)
	
	    # While the stop condition has not been met continue to train the embeddings.
	    while stop_condition.check_condition():
	
	        # Update the optimiser's timestep
	        self.optimiser.update_timestep()
	
	        # Shuffles the indices of the dataset.
	        dataloader.shuffle()
	
	        # A list of mean squared error for each batch in the epoch.
	        cur_epoch_loss = []
	
	        # Chunks the shuffled dataset indices into batches for training the graph embeddings.
	        for idx in range(0, len(dataloader), batch_size):
	
	            # Returns the actor and community indices within the batch.
	            actors_idx, comms_idx = dataloader.get_batch(idx)
	
	            # Extract batch for actor/comm embeddings.
	            w_a = self.W['a'][actors_idx, :]
	            w_c = self.W['c'][comms_idx, :]
	
	            # Calculate forward and backwards passes.
	            dL_dwa, dL_dwc, y_true, y_pred = self.loss_function.compute(w_a, w_c)
	
	            # Calculate gradient of batch.
	            actors_idx, dL_dwa = self.optimiser.calculate_batch_gradient(dL_dwa, actors_idx, key='a')
	            comms_idx, dL_dwc = self.optimiser.calculate_batch_gradient(dL_dwc, comms_idx, key='c')
	
	            # Update weights.
	            if self.update_actors:
	                scatter_add(self.W['a'], actors_idx, dL_dwa)
	
	            if self.update_comms:
	                scatter_add(self.W['c'], comms_idx, dL_dwc)
	
	            # Calculate the loss over the batch.
	            cur_epoch_loss.append(mean_squared_error(y_true, y_pred))
	
	        # Update stop condition with the average loss over the current epoch. Average taken over each batch in the
	        # epoch.
	        stop_condition.update(stats.mean(cur_epoch_loss))
	
	```

______


______

