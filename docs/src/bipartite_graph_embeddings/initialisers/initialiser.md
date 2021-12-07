## **Initialiser**`#!py3 class` { #Initialiser data-toc-label=Initialiser }



**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`generate_normal`](#generate_normal)
 - [`generate_uniform`](#generate_uniform)
 - [`populate`](#populate)
 - [`populate_normal`](#populate_normal)
 - [`populate_radial`](#populate_radial)
 - [`populate_uniform_sqrt`](#populate_uniform_sqrt)

### *Initialiser*.**__init__**`#!py3 (self, num_actors, num_comms, embedding_dim)` { #__init__ data-toc-label=__init__ }

Initialises the embeddings Initialiser class.

**Parameters**
> **num_actors:** ``int`` -- The number of vertices in the actor set.

> **num_comms:** ``int`` -- The number of vertices in the community set.

> **embedding_dim:** ``int`` -- The number of dimensions of each embedding.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, num_actors, num_comms, embedding_dim):
	    
	
	    # The number of actor vertices.
	    self.num_actors = num_actors
	
	    # The number of community vertices.
	    self.num_comms = num_comms
	
	    # The number of dimensions of each embedding.
	    self.embedding_dim = embedding_dim
	
	    # Options for populating weight initialisation.
	    self.init_method = {
	        'normal': self.populate_normal,
	        'uniform_sqrt': self.populate_uniform_sqrt,
	        'polar': self.populate_radial
	    }
	
	```

______

### *Initialiser*.**generate_normal**`#!py3 (self, num, mean=0, std=0.01)` { #generate_normal data-toc-label=generate_normal }

Initialises the embeddings from a normal distribution.

**Parameters**
> **mean:** ``int`` or ``float`` -- The mean value of the normal distribution.

> **std:** ``int`` or ``float`` -- The standard deviation of the normal distribution.

> **num:** ``int`` -- The number of embeddings to generate.

**Returns**
> **embeddings** ``array`` -- 2D array with dimensions [num, embedding_dim].


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_normal(self, num, mean=0, std=0.01):
	    
	    return xp.random.normal(mean, std, num * self.embedding_dim).reshape(num, self.embedding_dim)
	
	```

______

### *Initialiser*.**generate_uniform**`#!py3 (self, num, lower=-1.0, upper=1.0)` { #generate_uniform data-toc-label=generate_uniform }

Initialises the embeddings from a uniform distribution.

**Parameters**
> **lower:** ``int`` or ``float`` -- The lower bound of the uniform distribution.

> **upper:** ``int`` or ``float`` -- The upper bound of the uniform distribution.

> **num:** ``int`` -- The number of embeddings to generate.

**Returns**
> **embeddings** ``array`` -- 2D array with dimensions [num, embedding_dim].


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def generate_uniform(self, num, lower=-1.0, upper=1.0):
	    
	    return xp.random.uniform(lower, upper, num * self.embedding_dim).reshape(num, self.embedding_dim)
	
	```

______

### *Initialiser*.**populate**`#!py3 (self, init)` { #populate data-toc-label=populate }

Decides which embedding initialisation method to use.

!!! error "Depreciating"
    This method will likely be depreciated when an optimal initialisation method is determined.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def populate(self, init):
	    
	    return self.init_method[init]()
	
	```

______

### *Initialiser*.**populate_normal**`#!py3 (self, mean=0, std=0.01)` { #populate_normal data-toc-label=populate_normal }

Initialises the starting positions for the actor and community vertices. Starting position are chosen from a
normal distribution with chosen parameters (mean, std).

**Parameters**
> **mean:** ``int`` or ``float`` -- The mean value of the normal distribution.

> **std:** ``int`` or ``float`` -- The standard deviation of the normal distribution.

**Returns**
> **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def populate_normal(self, mean=0, std=0.01):
	    
	
	    W = {
	        'a': self.generate_normal(self.num_actors, mean, std),
	        'c': self.generate_normal(self.num_comms, mean, std)
	    }
	
	    return W
	
	```

______

### *Initialiser*.**populate_radial**`#!py3 (self)` { #populate_radial data-toc-label=populate_radial }

Initialises the starting position for the actor and community vertices. Starting positions are chosen from a
uniform distribution between -pi and pi.

**Returns**
> **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def populate_radial(self):
	    
	
	    W = {
	        'a': self.generate_normal(self.num_actors, 0, xp.pi),
	        'c': self.generate_normal(self.num_comms, 0, xp.pi)
	    }
	
	    return W
	
	```

______

### *Initialiser*.**populate_uniform_sqrt**`#!py3 (self)` { #populate_uniform_sqrt data-toc-label=populate_uniform_sqrt }

Initialises the starting positions for the actor and community vertices. Starting position are chosen from a
uniform distribution between [-1/sqrt(n), 1/sqrt(n)] where n is the number of vertices in the vertex's set.

**Returns**
> **W:** ``dictionary`` -- A dictionary containing the embeddings for the actor and community vertices.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def populate_uniform_sqrt(self):
	    
	
	    W = {
	        'a': self.generate_uniform(self.num_actors, -1/sqrt(self.num_actors), 1/sqrt(self.num_actors)),
	        'c': self.generate_uniform(self.num_comms, -1/sqrt(self.num_comms), 1/sqrt(self.num_comms))
	    }
	
	    return W
	
	```

______


______

