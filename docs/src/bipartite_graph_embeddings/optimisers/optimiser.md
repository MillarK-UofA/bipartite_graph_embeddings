## **Optimiser**`#!py3 class` { #Optimiser data-toc-label=Optimiser }



**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`calculate_batch_gradient`](#calculate_batch_gradient)
 - [`replace_with_unique`](#replace_with_unique)
 - [`update_timestep`](#update_timestep)

### *Optimiser*.**__init__**`#!py3 (self, alpha, num_actors, num_comms, embedding_dim)` { #__init__ data-toc-label=__init__ }

Initialises the Optimiser class.

**Parameters**
> **alpha:** ``float`` -- The initial learning rate.

> **num_actors:** ``int`` -- The number of vertices in the actor set.

> **num_comms:** ``int`` -- The number of vertices in the community set.

> **embedding_dim:** ``int`` -- The number of dimensions of each embedding.

!!! note "num_actors/num_comms"
    ``num_actors`` and ``num_comms`` are not required for the SGD class; however, they are still defined to keep
    the parameters consistent with the optimiser classes.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, alpha, num_actors, num_comms, embedding_dim):
	
	    
	
	    # Learning Rate
	    self.alpha = alpha
	
	    # Graph parameters
	    self.num_actors = num_actors
	    self.num_comms = num_comms
	    self.embedding_dim = embedding_dim
	
	    # timestep and alpha at the initial timestep.
	    self.t = 1
	    self.alpha_t = self.alpha
	
	```

______

### *Optimiser*.**calculate_batch_gradient**`#!py3 (self, grad, indices, key)` { #calculate_batch_gradient data-toc-label=calculate_batch_gradient }

Calculates the gradient over a batch.

**Parameters**
> **grad:** ``array`` -- Individual gradients.

> **indices:** ``array`` -- The vertices corresponding to the individual gradients.

> **key:** ``string`` -- Whether or this gradient affects the actor or community set. key='a' if this gradient
is over the actor set; else key='c'.

!!! note
    Overridden by child classes.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def calculate_batch_gradient(self, grad, indices, key):
	    
	    pass
	
	```

______

### *Optimiser*.**replace_with_unique**`#!py3 (array, unique)` { #replace_with_unique data-toc-label=replace_with_unique }

Replace elements in ``array`` with their index in ``unique``.

> **array:** ``array`` -- array of elements to replace with their index in unique.

> **unique:** ``array`` -- A list of unique elements from ``array``.

**Returns**
> **unique_array:** ``array`` -- ``array`` where the value of elements have been replace by their index in the
``unique`` array.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@staticmethod
	def replace_with_unique(array, unique):
	    
	    sidx = unique.argsort()
	    return sidx[xp.searchsorted(unique, array, sorter=sidx)]
	
	```

______

### *Optimiser*.**update_timestep**`#!py3 (self)` { #update_timestep data-toc-label=update_timestep }

Updates the time step 't' and returns the corresponding alpha for this given time step.

!!! note
    Overridden by child classes.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def update_timestep(self):
	    
	    pass
	
	```

______


______

