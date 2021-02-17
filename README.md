Description
***********

The notebook "illustrations.ipynb" implements the experiments presented in https://arxiv.org/abs/2010.08356. It also contains a small example of filter selection. In order to reproduce the full table of results given by filter selection presented in the article, one should launch the script "launch_optim_filters.sh" instead. If you are doing this, do not forget to update your local path in line 117 of optim_filters.py.

Dependencies
************

Our code is based on the Gudhi library (http://gudhi.gforge.inria.fr/python/latest/), which can be installed, e.g., by running "conda install gudhi" in an Anaconda environment. Our code also depends on Tensorflow 2.4.1. 

Data sets
*********

Most external data sets are available in the "data" repository. The MNIST data set is available in Tensorflow 2.4.1. Graph data sets have to be extracted from graphs.zip.
