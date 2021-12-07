<div align="center">
<h1 style="margin-top:0; margin-bottom: 0">Biparite Graph Embeddings (BGE)</h1>
<h3>Kyle Millar (kyle.millar@adelaide.edu.au)<br>(v3.0) 02/12/2021</h3>
</div>

- - - -
 
<h2> Project Summary </h2>
<div style="text-align: justify">
This repo contains all the necessary tools to utilise bipartite graph embeddings (BGE)
</div>
<br>

Documentation has been created using the [mkdocs](https://pypi.org/project/mkdocs/) python package. To view documentation:
1. Install mkdocs and extras: 
    
   -  ``pip install mkdocs``
    
   - ``pip install mkdocs-material-extensions``
    
2. From "docs" directory: ``python -m mkdocs serve``
3. Navigate to [localhost:8000](http://localhost:8000) in your preferred browser.


<h2> Directory Structure </h2>

| Directory                      | Short Description                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------------------------|
| ./docs                           | Contains documentation of all scripts (mkdocs)                                                      |
| ./lib/bipartite_graph_embeddings | Scripts to train bipartite graph embeddings (BGE).                                                  |
| ./lib/common                     | Common functions used throughout the codebase.                                                      |