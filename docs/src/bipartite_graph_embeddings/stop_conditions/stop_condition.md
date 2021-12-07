## **StopCondition**`#!py3 class` { #StopCondition data-toc-label=StopCondition }

Defines the stopping conditions used in Bipartite Graph Embeddings (BGE). Also provides feedback to the user about
current progress.

**class functions & static methods:** 

 - [`__init__`](#__init__)
 - [`check_condition`](#check_condition)
 - [`improving_loss_condition`](#improving_loss_condition)
 - [`max_epoch_condition`](#max_epoch_condition)
 - [`update`](#update)

### *StopCondition*.**__init__**`#!py3 (self, epoch_sample_size, max_epochs)` { #__init__ data-toc-label=__init__ }

Initialises the StopCondition class.

**Parameters**
> **epoch_sample_size:** ``int`` -- The number of epochs to run before checking if the stop condition has been
met. Averages results of all epochs in the sample size.

> **max_epochs:** ``int`` or ``None`` -- The number of epochs to run. If ``None``, training is will continue
until the stop condition is met.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, epoch_sample_size, max_epochs):
	    
	
	    # Starting Conditions
	    self.prev_epoch_loss = 1
	    self.delta_epoch_loss = -1
	    self.epoch_idx = 0
	    self.epoch_loss = []
	    self.max_epochs = max_epochs
	
	    # Number of epochs to run before checking stop condition.
	    self.epoch_sample_size = epoch_sample_size
	
	    # Progress Bar.
	    self.bar = tqdm(total=max_epochs)
	
	    # Stop condition.
	    # if max_epochs is set, the stop condition is set to run that many epochs. Else, the stopping condition is set
	    # such that training will stop when the loss function does not improve from the previous sample of epochs.
	    self.condition = self.max_epoch_condition if max_epochs > 0 else self.improving_loss_condition
	
	```

______

### *StopCondition*.**check_condition**`#!py3 (self)` { #check_condition data-toc-label=check_condition }

Checks whether the stopping condition has been met.

**Returns**
> **stop_condition:** ``bool`` -- True if the stop condition has not been reached.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def check_condition(self):
	    
	    if self.condition():
	        return True
	    else:
	        self.bar.close()
	        return False
	
	```

______

### *StopCondition*.**improving_loss_condition**`#!py3 (self)` { #improving_loss_condition data-toc-label=improving_loss_condition }

Defines the improving_loss stop condition. Training is stopped when the loss function does not improve over the
previous ``epoch_sample_size``.

**Returns**
> **stop_condition:** ``bool`` -- True if the stop condition has not been reached.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def improving_loss_condition(self):
	    
	    return self.delta_epoch_loss <= 0
	
	```

______

### *StopCondition*.**max_epoch_condition**`#!py3 (self)` { #max_epoch_condition data-toc-label=max_epoch_condition }

Defines the max_epoch stop condition.

**Returns**
> **stop_condition:** ``bool`` -- True if the stop condition has not been reached.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def max_epoch_condition(self):
	    
	    return self.epoch_idx < self.max_epochs
	
	```

______

### *StopCondition*.**update**`#!py3 (self, cur_epoch_loss)` { #update data-toc-label=update }

Updates the stopping condition with results from the current epoch.

**Parameters**
> **cur_epoch_loss:** ``float`` -- The squared mean error of the current epoch.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def update(self, cur_epoch_loss):
	    
	
	    # Record total loss of the current epoch.
	    self.epoch_loss.append(cur_epoch_loss)
	
	    self.bar.update(1)
	    self.epoch_idx += 1
	
	    if len(self.epoch_loss) == self.epoch_sample_size:
	
	        # Calculate the mean batch loss over the epoch.
	        avg_epoch_loss = stats.mean(self.epoch_loss)
	
	        # Calculate the change in loss from the previous epoch.
	        self.delta_epoch_loss = avg_epoch_loss - self.prev_epoch_loss
	
	        # Update previous epoch loss with current epoch loss.
	        self.prev_epoch_loss = avg_epoch_loss
	
	        # Update bar chart.
	        self.bar.set_description("Epoch: {:,} Loss: {:.6f}, delta loss {:.6f}"
	                                 .format(self.epoch_idx, avg_epoch_loss, self.delta_epoch_loss))
	
	        # Reset epoch loss for the next sample.
	        self.epoch_loss = []
	
	```

______


______

