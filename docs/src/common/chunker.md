
Functions that split up a series of elements into multiple chunks. Useful for splitting a list/dict into multiple chunks
for multiprocessing.

______

### **chunk_dict**`#!py3 (d, n)` { #chunk_dict data-toc-label=chunk_dict }

Splits a dictionary into a series of chunks of length 'n'. If the dict does not divide evenly into chunks of
length 'n' the last chunk will have fewer elements.

**Parameters**
> **d:** ``dict`` -- The dict to split into multiple check.

> **n:** ``int`` -- The length of the chunk.

**Returns**
> **results:** ``list`` -- A list of chunks from the original dict.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def chunk_dict(d, n):
	    
	
	    l = list(d.items())
	    return chunk_list(l, n)
	
	```

______

### **chunk_list**`#!py3 (l, n)` { #chunk_list data-toc-label=chunk_list }

Splits a list into a series of chunks of length 'n'. If the list does not divide evenly into chunks of length 'n'
the last chunk will have fewer elements.

**Parameters**
> **l:** ``list`` -- The list to split into multiple check.

> **n:** ``int`` -- The length of the chunk.

**Returns**
> **results:** ``list`` -- A list of chunks from the original list.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def chunk_list(l, n):
	    
	    return list(enumerate_chunk_list(l, n))
	
	```

______

### **enumerate_chunk_list**`#!py3 (l, n)` { #enumerate_chunk_list data-toc-label=enumerate_chunk_list }

Iterates through a list and yields 'n' sized chunks. If the list does not divide evenly into chunks of length 'n'
the last chunk will have fewer elements.

**Parameters**
> **l:** ``list`` -- The list to split into multiple check.

> **n:** ``int`` -- The length of the chunk.


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def enumerate_chunk_list(l, n):
	    
	    for i in range(0, len(l), n):
	        yield l[i:i + n]
	
	```

______

