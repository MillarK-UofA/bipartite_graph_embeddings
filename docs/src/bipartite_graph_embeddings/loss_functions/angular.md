
Defines the Angular Loss Function.

______

## **Angular**`#!py3 class` { #Angular data-toc-label=Angular }



**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`activation_exp`](#activation_exp)
 - [`activation_none`](#activation_none)
 - [`activation_sigmoid`](#activation_sigmoid)
 - [`activation_tanh`](#activation_tanh)
 - [`compute`](#compute)
 - [`compute_y_true`](#compute_y_true)

### *Angular*.**__init__**`#!py3 (self, activation='exp')` { #__init__ data-toc-label=__init__ }

Initialises the Angular Loss Function class.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, activation='exp'):
	    
	    super().__init__(activation)
	
	```

______

### *Angular*.**activation_exp**`#!py3 (self, z)` { #activation_exp data-toc-label=activation_exp }




??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def activation_exp(self, z):
	
	    diag = xp.identity(z.shape[-1])
	    off_diag = xp.subtract(xp.ones_like(diag), diag)
	
	    y_pred = xp.exp(z) * diag + xp.exp(-z) * off_diag
	
	    # Get the true matrix.
	    y_true = self.compute_y_true(y_pred, lims=(xp.exp(1), xp.exp(1)))
	
	    # calculate the difference between y_true and y_pred matrices.
	    loss = xp.subtract(y_true, y_pred)
	
	    dy_dz = y_pred * diag - y_pred * off_diag
	
	    dL_dz = xp.multiply(loss, dy_dz)
	
	    return y_pred, y_true, dL_dz
	
	```

______

### *Angular*.**activation_none**`#!py3 (self, y_pred)` { #activation_none data-toc-label=activation_none }

No activation function.

Computes the difference between the true and predicted matrices.

!!! note "Loss"
    Technically the loss should be the squared difference of the true and prediction matrices; however, in
    practice, the difference alone does not significant change the training performance while removing some
    complexity.

!!! warning "y_true Configuration"
    The configuration for generating the true matrix for the ``activation_none`` function is setup to work with
    the ``cosine`` loss function. Due to the limits of the cosine function [-1, 1], a negative example is set to
     -1 and a positive samples is set to 1.

**Parameters**
> **y_pred:** ``array`` -- The prediction matrix.

**Returns**
> **y_pred:** ``array`` -- The prediction matrix.

> **y_true:** ``array`` -- The true matrix.

> **loss:** ``array`` -- The difference between the true and predicted matrices.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def activation_none(self, y_pred):
	    
	
	    # Get the true matrix.
	    y_true = self.compute_y_true(y_pred, lims=(-1, 1))
	
	    # calculate the difference between y_true and y_pred matrices.
	    dL_dy = xp.subtract(y_true, y_pred)
	
	    return y_pred, y_true, dL_dy
	
	```

______

### *Angular*.**activation_sigmoid**`#!py3 (self, z)` { #activation_sigmoid data-toc-label=activation_sigmoid }

Computes the forward and backward passes of the sigmoid activation function.

**Parameters**
> **z:** ``array`` -- The input to the sigmoid activation function.

**Returns**
> **y_pred:** ``array`` -- The prediction matrix.

> **y_true:** ``array`` -- The true matrix.

> **dL_dz:** ``array`` -- The change in loss with respect to the change in z.

!!! note "Loss"
    Technically the loss should be the squared difference of the true and prediction matrices; however, in
    practice, the difference alone does not significant change the training performance while removing some
    complexity.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def activation_sigmoid(self, z):
	    
	
	    # Compute the elementwise sigmoid of the prediction matrix.
	    y_pred = sigmoid(z)
	
	    # Get the true matrix.
	    y_true = self.compute_y_true(y_pred)
	
	    # calculate the difference between y_true and y_pred matrices.
	    loss = xp.subtract(y_true, y_pred)
	
	    # Change in y_pred with respect to z.
	    dy_dz = xp.subtract(y_pred, xp.power(y_pred, 2))
	
	    # Calculate the change in loss with respect to the change in z. (Chain Rule).
	    dL_dz = xp.multiply(loss, dy_dz)
	
	    return y_pred, y_true, dL_dz
	
	```

______

### *Angular*.**activation_tanh**`#!py3 (self, z)` { #activation_tanh data-toc-label=activation_tanh }

Computes the forward and backward passes of the sigmoid activation function.

**Parameters**
> **z:** ``array`` -- The input to the sigmoid activation function.

**Returns**
> **y_pred:** ``array`` -- The prediction matrix.

> **y_true:** ``array`` -- The true matrix.

> **dL_dz:** ``array`` -- The change in loss with respect to the change in z.

!!! note "Loss"
    Technically the loss should be the squared difference of the true and prediction matrices; however, in
    practice, the difference alone does not significant change the training performance while removing some
    complexity.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def activation_tanh(self, z):
	    
	
	    y_pred = xp.tanh(z)
	
	    # Get the true matrix.
	    y_true = self.compute_y_true(y_pred, lims=(-1, 1))
	
	    # calculate the difference between y_true and y_pred matrices.
	    loss = xp.subtract(y_true, y_pred)
	
	    # Change in y_pred with respect to z.
	    dy_dz = xp.ones_like(y_pred) - xp.power(xp.tanh(y_pred), 2)
	
	    # Calculate the change in loss with respect to the change in z. (Chain Rule).
	    dL_dz = xp.multiply(loss, dy_dz)
	
	    return y_pred, y_true, dL_dz
	
	```

______

### *Angular*.**compute**`#!py3 (self, w_a, w_c)` { #compute data-toc-label=compute }

Computes the change in actor and community weights to optimise the cosine of the difference between to angles.

**Parameters**
> **w_a:** ``array`` -- The actor embeddings to be updated.

> **w_c:** ``array`` -- The community embeddings to be updated.

**Returns**
> **dL_dwa:** ``array`` -- The change in loss function with respect to the change in actor embeddings.

> **dL_dwc:** ``array`` -- The change in loss function with respect to the change in community embeddings.

> **y_true:** ``array`` -- The true matrix.

> **y_ture:** `` array`` -- The prediction matrix.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def compute(self, w_a, w_c):
	    
	
	    # Reshape embeddings.
	    w_a = w_a.reshape((w_a.shape[0], w_a.shape[2], w_a.shape[1], 1))
	    w_c = w_c.reshape((w_c.shape[0], w_c.shape[2], 1, w_c.shape[1]))
	
	    # - Forward Pass --------------------------------------------------------------------------------------------- #
	    # Calculate the cosine of the difference between the actor and community embeddings.
	    z = xp.cos(xp.subtract(w_a, w_c))
	
	    # Passes z through sigmoid activation function.
	    y_pred, y_true, dL_dz = self.activation(z)
	    # ------------------------------------------------------------------------------------------------------------ #
	
	    # Calculate change in z with respect to w_a, dz_dwa
	    # Note change in z with respect to w_c is simply the negative of dz_dwa.
	    dz_dwa = xp.matmul(-xp.sin(w_a), xp.cos(w_c)) + xp.matmul(xp.cos(w_a), xp.sin(w_c))
	
	    dz_dwa = xp.sign(dz_dwa)
	
	    # Calculate change in loss with respect to w_a.
	    k = xp.multiply(dL_dz, dz_dwa)
	
	    # Calculate change in loss with respect to w_a.
	    dL_dwa = xp.transpose(xp.sum(k, axis=3), axes=(0, 2, 1))
	
	    # Calculate change in loss with respect to w_c.
	    dL_dwc = xp.transpose(xp.sum(-k, axis=2), axes=(0, 2, 1))
	
	    return dL_dwa, dL_dwc, y_true, y_pred
	
	```

______

### *Angular*.**compute_y_true**`#!py3 (y_pred, lims=(0, 1))` { #compute_y_true data-toc-label=compute_y_true }

Returns the true matrix.

**Parameters**
> **y_pred:** ``array`` -- The prediction matrix. Used to define the shape of the true matrix.

> **lims:** ``tuple`` -- Defines the positive and negative example values. Default,
``negative_exmaples=0`` and ``positive_examples=1``.

**Returns**
> **y_true:** ``array`` -- The true matrix.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@staticmethod
	def compute_y_true(y_pred, lims=(0, 1)):
	    
	
	    s0, _, s2 = y_pred.shape
	    y_true = xp.full_like(y_pred, fill_value=lims[0])
	    y_true.reshape(s0, -1)[:, ::s2 + 1] = lims[1]
	
	    return y_true
	
	```

______


______

