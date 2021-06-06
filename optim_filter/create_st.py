import os
import random
import time
import itertools  
import numpy                 as np
import tensorflow            as tf
import pandas                as pd
import gudhi                 as gd
import gudhi.representations as sktda
import sys

path = '/home/mcarrier/Github/difftda/'

sys.path.append(path)
from difftda               import *

from gudhi.representations.vector_methods import Atol as atol
from gudhi.representations.kernel_methods import SlicedWassersteinKernel as swk

from gudhi.wasserstein         import wasserstein_distance
from gudhi.representations     import pairwise_persistence_diagram_distances as ppdd
from scipy.linalg              import expm
from scipy.io                  import loadmat
from scipy.sparse              import csgraph
from scipy.linalg              import eigh
from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.metrics           import pairwise_distances, accuracy_score
from sklearn.manifold          import MDS, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing     import MinMaxScaler, Normalizer, LabelEncoder
from sklearn.pipeline          import Pipeline
from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.model_selection   import GridSearchCV, KFold, StratifiedKFold
from sklearn.cluster           import KMeans

from spektral.datasets import TUDataset

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')

dataset = sys.argv[1]

if (dataset[:2] == 'vs') or (dataset == 'all'):
    data_type = 'MNIST'
else:
    data_type = 'GRAPHS'

use_spektral = True

if data_type == 'GRAPHS':

    path = path + 'data/graphs/' + dataset + '/mat/'
    
    if use_spektral:

        data = TUDataset(dataset)
        for i in range(len(data)):
            stfile = open(path + 'SPK' + str(i) + '_st.txt', 'w')
            A = np.array(data[i].a.todense())
            num_vertices = A.shape[0]
            for v in range(num_vertices):
                stfile.write(str(v) + ' \n')
            idxs = np.argwhere(A > 0)
            for v in range(len(idxs)):
                stfile.write(str(idxs[v,0]) + ' ' + str(idxs[v,1]) + ' \n')
            stfile.close()

    else:

        pad_size = 50
        Cinit = np.ones([1,pad_size])
        graph_names = np.array(os.listdir(path))
        for graph in graph_names:
            if graph[-4:] == '.mat':
                stfile = open(path + graph + '_st.txt', 'w')
                A = np.array(loadmat(path + graph)['A'], dtype=np.float32)
                num_vertices = A.shape[0]
                for i in range(num_vertices):
                    stfile.write(str(i) + ' \n')
                idxs = np.argwhere(A > 0)
                for i in range(len(idxs)):
                    stfile.write(str(idxs[i,0]) + ' ' + str(idxs[i,1]) + ' \n')
                stfile.close()

