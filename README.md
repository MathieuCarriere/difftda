Description
***********

The notebook `illustrations.ipynb` implements the experiments presented in the article "Optimizing persistent homology based functions". The experiments on filter selection can be run with the code in `optim_filter/` folder. This code was meant to be run on cluster, so if you do not use a cluster, please remove the `oarsub -S` and the python specific Python environment loadings (`module load ...`, `eval ...` and `conda activate ...`) in the `.sh` files, and update the variable `path` (at the beginning to the `.py` files) to the local Github repository path on your machine.

Dependencies
************

Our code is based on the Gudhi library (http://gudhi.gforge.inria.fr/python/latest/), which can be installed, e.g., by running `conda install gudhi` in an Anaconda environment. Our code also depends on Tensorflow 2.4.1. 

Data sets
*********

Most external data sets are available in the `data` repository. The MNIST data set is available in Tensorflow 2.4.1. Graph data sets have to be extracted from `graphs.zip`.
