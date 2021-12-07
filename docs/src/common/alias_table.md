
Defines the alias tables used in bipartite graph embeddings.

Code in this script have been sourced from the Laboratory for Intelligent Probabilistic Systems:
https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

______

### **alias_draw**`#!py3 (J, q)` { #alias_draw data-toc-label=alias_draw }

Draws a sample from the sampling table.

**Parameters**
> **J:** ``np.array`` -- Large distribution

> **q:** ``np.array`` -- Small distribution

**Returns**
> **sample:** ``int`` -- the drawn sample


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def alias_draw(J, q):
	    
	
	    K = len(J)
	
	    # Draw from the overall uniform mixture.
	    kk = int(np.floor(np.random.rand() * K))
	
	    # Draw from the binary mixture, either keeping the
	    # small one, or choosing the associated larger one.
	    if np.random.rand() < q[kk]:
	        return kk
	    else:
	        return J[kk]
	
	```

______

### **alias_setup**`#!py3 (probs)` { #alias_setup data-toc-label=alias_setup }

Defines the alias table from an input distribution.

**Parameters**
> **probs:** ``list`` -- The specified distribution.

**Returns**
> **J:** ``np.array`` -- Large distribution

> **q:** ``np.array`` -- Small distribution


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def alias_setup(probs):
	    
	
	    K = len(probs)
	    q = np.zeros(K)
	    J = np.zeros(K, dtype=np.int)
	
	    # Sort the data into the outcomes with probabilities
	    # that are larger and smaller than 1/K.
	    smaller = []
	    larger = []
	    for kk, prob in enumerate(probs):
	        q[kk] = K * prob
	        if q[kk] < 1.0:
	            smaller.append(kk)
	        else:
	            larger.append(kk)
	
	    # Loop though and create little binary mixtures that
	    # appropriately allocate the larger outcomes over the
	    # overall uniform mixture.
	    while len(smaller) > 0 and len(larger) > 0:
	        small = smaller.pop()
	        large = larger.pop()
	
	        J[small] = large
	        q[large] = q[large] - (1.0 - q[small])
	
	        if q[large] < 1.0:
	            smaller.append(large)
	        else:
	            larger.append(large)
	
	    return J, q
	
	```

______

