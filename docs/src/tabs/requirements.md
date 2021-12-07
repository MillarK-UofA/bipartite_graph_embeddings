This page contains all third-party software packages required to run the scripts contained in the AG codebase.

##Software Requirements:
| Software Support | Validated Version | Short Description |
| -------- | ----------------- | ----------------- |
| [Python](https://www.python.org/downloads/release/python-360/) | 3.8.1 | Python is the default programming language used throughout the codebase. |
| [PostgreSQL](https://www.postgresql.org/) | 11.5 | Opensource database management system. You will have to create two data bases within PostgreSQL: "affiliation_graphs" and "common". |
| [CUDA](https://developer.nvidia.com/cuda-zone) | 11.2 | CUDA is required to run cupy python libraries. (*Optional* but extremely recommended) |

    
!!! warning "CUDA versions"
    The version of CUDA you have installed will dictate what version of cupy you should install. 
    If unsure, please use the versions specified on this page for [cupy](https://docs.cupy.dev/en/stable/install.html).


##Python Modules

| Python Module                                             | Validated Version | Short Description                                                      |
|-----------------------------------------------------------| ------- |----------------------------------------------------------------------------------|
| [networkx](https://pypi.org/project/networkx/)            | v2.6.3  | Creates, manipulates, visualises, and provides analysis tools for graphs.        |
| [NumPy](https://pypi.org/project/numpy/)                  | v1.21.4 | Efficient array computations.                                                    |
| [Pandas](https://pypi.org/project/pandas/)                | v1.3.4  | Creates data structures for easy data processing.                                |
| [tqdm](https://pypi.org/project/tqdm/)                    | v4.62.3 | Provides a visual progressbar within the command line.                           |
| [scipy](https://pypi.org/project/scipy/)                  | v1.7.3  | Functions for mathematics, science, and engineering.                             |
| [scikit-learn](https://pypi.org/project/scikit-learn/)    | v1.0.1  | Extension of SciPy for common machine learning functionality.                    |
| [mkdocs](https://pypi.org/project/mkdocs/)                | v1.2.3  | Generates markdown documentation.                                                |
| [mkdocs-material-extensions](https://pypi.org/project/mkdocs-material-extensions/) | v1.0.3 | Extra functionality for markdown documentation.            |
| [prettytable](https://pypi.org/project/prettytable/)      | v2.4.0  | A simply way of displaying tables within std output.                             |
| [cupy](https://pypi.org/project/cupy/)                    | v9.6.0  (cupy-cuda112) | Performs numpy-like operations on the GPU. Requires a CUDA enabled GPU (nVIDIA). |


!!! note "cupy"
    If cupy is not installed, numpy will be used instead. See [cupy_support](http://localhost:8000/common/cupy_support.md)
