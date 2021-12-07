
cupy_support has been designed to provide a common interface for cupy and numpy operations. While the interface
between numpy and cupy are very similar there are edges cases where they have to be called differently. This is
mainly due to the current development stage of cupy.

A common interface between cupy and numpy was required to achieve a more flexible design. Cupy can greatly speed up
operations on large matrices; however, it requires a GPU with CUDA support (i.e., a Nvidia GPU). For systems without a
Nvidia GPU, Numpy will be used. While numpy is still usable, for cupy can see 10x-60x time improvement for large graphs.

!!! note "Small Matrices"
    Numpy is actually faster on small matrices. This is to due to the overhead of setting the operations up on the GPU.
    For operations with less than ~10^4 elements numpy will likely be faster.

______

### **mean_squared_error**`#!py3 (y_true, y_pred)` { #mean_squared_error data-toc-label=mean_squared_error }

Calculates the mean squared error between matrices ``y_true`` and ``y_pred``. This function has been defined to
provide a common mean_squared_error function for both cupy and Numpy.

!!! note "Numpy MSE"
    Numpy does have a mse error function in the form of sklearn.metrics.mean_squared_error; however, it does not
    work for 3-dimensional arrays.

**Parameters**
> **y_true:** ``array`` -- True results.

> **y_pred:** ``array`` -- Predicted results.

**Returns**
> **mse:** ``float`` -- Mean squared error between arrays - ``y_true`` and ``y_pred``.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def mean_squared_error(y_true, y_pred):
	    
	    return float(xp.mean(xp.square(y_true - y_pred)))
	
	```

______

### **sigmoid**`#!py3 (x, b=1)` { #sigmoid data-toc-label=sigmoid }

Passes input through the sigmoid function. Bounds input between [0, 1]. This function has been defined as cupy and
numpy both use unique function calls for the sigmoid function.

**Parameters**
> **x:** ``array`` -- an n-dimensional array for which to compute the element-wise sigmoid upon.

> **b:** ``float`` -- A variable used to modify the slope of the sigmoid function.

**Returns**
> **x:** ``array`` -- The result of applying an element-wise sigmoid to the input array.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def sigmoid(x, b=1):
	    
	    ones = xp.ones(shape=x.shape)
	    return xp.divide(ones, xp.add(ones, xp.exp(b * xp.negative(x))))
	
	```

______

### **xp_round**`#!py3 (array, decimals=6)` { #xp_round data-toc-label=xp_round }

Round the elements of an array to a given number of decimals. This has been defined as cupy uses ``around`` as its
rounding function and numpy uses ``round``.

**Parameters**
> **array:** ``array`` -- The array to be rounded.

> **decimals:** ``int`` -- The number of decimals to round each element to.

**Returns**
> **array:** ``array`` -- The rounded array.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def xp_round(array, decimals=6):
	    
	
	    if _cupy_available:
	        return xp.around(array, decimals)
	    else:
	        return xp.round(array, decimals)
	
	```

______

