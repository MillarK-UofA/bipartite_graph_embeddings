## **Adam**`#!py3 class` { #Adam data-toc-label=Adam }



**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`calculate_batch_gradient`](#calculate_batch_gradient)
 - [`replace_with_unique`](#replace_with_unique)
 - [`update_timestep`](#update_timestep)

### *Adam*.**__init__**`#!py3 (self, alpha, num_actors, num_comms, embedding_dim)` { #__init__ data-toc-label=__init__ }

Initialises the Adam optimiser class.

**Parameters**
> **alpha:** ``float`` -- The initial learning rate.

> **num_actors:** ``int`` -- The number of vertices in the actor set.

> **num_comms:** ``int`` -- The number of vertices in the community set.

> **embedding_dim:** ``int`` -- The number of dimensions of each embedding.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, alpha, num_actors, num_comms, embedding_dim):
	    
	    super().__init__(alpha, num_actors, num_comms, embedding_dim)
	
	    # Adam Optimisation Parameters
	    # These are rarely changed in practise.
	    self.beta_1 = 0.9
	    self.beta_2 = 0.999
	    self.epsilon = pow(10, -8)
	
	    # First order momentum
	    self.M = {
	        'a': xp.zeros(shape=(self.num_actors, self.embedding_dim), dtype=float),
	        'c': xp.zeros(shape=(self.num_comms, self.embedding_dim), dtype=float)
	    }
	
	    # Second order momentum
	    self.V = {
	        'a': xp.zeros(shape=(self.num_actors, self.embedding_dim), dtype=float),
	        'c': xp.zeros(shape=(self.num_comms, self.embedding_dim), dtype=float)
	    }
	
	```

______

### *Adam*.**calculate_batch_gradient**`#!py3 (self, grad, indices, key)` { #calculate_batch_gradient data-toc-label=calculate_batch_gradient }

Calculates the gradient over a batch using the Adam optimisation technique.

**Parameters**
> **grad:** ``array`` -- Individual gradients.

> **indices:** ``array`` -- The vertices corresponding to the individual gradients.

> **key:** ``string`` -- Whether or this gradient affects the actor or community set. key='a' if this gradient
is over the actor set; else key='c'.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def calculate_batch_gradient(self, grad, indices, key):
	    
	
	    # Count the number of changes for each modified vertex.
	    unique, counts = xp.unique(indices, return_counts=True)
	
	    # Replace indices with their index in the unique array.
	    indices = self.replace_with_unique(indices, unique)
	
	    # Reshape counts to simplify subsequent operations.
	    counts = counts.reshape(-1, 1)
	
	    # Create a zero matrix for the first and second momentum of each vertex to be updated.
	    m_new = xp.zeros((unique.shape[0], self.embedding_dim), dtype=float)
	    v_new = xp.zeros((unique.shape[0], self.embedding_dim), dtype=float)
	
	    # Add all gradients for each respective gradient.
	    scatter_add(m_new, indices, grad)
	    scatter_add(v_new, indices, xp.square(grad))
	
	    # Calculate the change in the first and second order momentum
	    m_new = self.beta_1 * self.M[key][unique] + (1 - self.beta_1) * xp.divide(m_new, counts)
	    v_new = self.beta_2 * self.V[key][unique] + (1 - self.beta_2) * xp.divide(v_new, counts)
	
	    # Calculate the new gradient.
	    dL = self.alpha_t * xp.divide(m_new, xp.add(xp.sqrt(v_new), self.epsilon))
	
	    # Update momentum matrices.
	    self.M[key][unique] = m_new
	    self.V[key][unique] = v_new
	
	    return unique, dL
	
	```

______

### *Adam*.**replace_with_unique**`#!py3 (array, unique)` { #replace_with_unique data-toc-label=replace_with_unique }

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

### *Adam*.**update_timestep**`#!py3 (self)` { #update_timestep data-toc-label=update_timestep }

Updates the time step 't' and returns the corresponding alpha for this given time step.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def update_timestep(self):
	    
	    self.t += 1
	    self.alpha_t = self.alpha * sqrt(1-pow(self.beta_2, self.t)) / (1-pow(self.beta_1, self.t))
	
	```

______


______

