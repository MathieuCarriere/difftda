import random
import time
import itertools  
import numpy                 as np
import tensorflow            as tf
import matplotlib.pyplot     as plt
import pandas                as pd
import gudhi                 as gd
import gudhi.representations as sktda
import sys
import os

path = '/home/mcarrier/Github/difftda/'

import sys
sys.path.append(path)

from difftda                              import *
from gudhi.representations.vector_methods import Atol as atol
from gudhi.representations.kernel_methods import SlicedWassersteinKernel as swk
from gudhi.wasserstein                    import wasserstein_distance
from gudhi.representations                import pairwise_persistence_diagram_distances as ppdd
from mpl_toolkits.mplot3d                 import Axes3D
from scipy.linalg                         import expm
from scipy.io                             import loadmat
from scipy.sparse                         import csgraph
from scipy.linalg                         import eigh
from sklearn.base                         import BaseEstimator, TransformerMixin
from sklearn.metrics                      import pairwise_distances, accuracy_score
from sklearn.manifold                     import MDS, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing                import MinMaxScaler, Normalizer, LabelEncoder
from sklearn.pipeline                     import Pipeline, FeatureUnion
from sklearn.svm                          import SVC
from sklearn.ensemble                     import RandomForestClassifier
from sklearn.neighbors                    import KNeighborsClassifier
from sklearn.model_selection              import GridSearchCV, KFold, StratifiedKFold
from sklearn.cluster                      import KMeans

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
from spektral.utils.convolution import gcn_filter
from spektral.datasets import TUDataset
from spektral.layers import GINConv, GlobalAvgPool
from spektral.data.loaders import SingleLoader

class FiltrationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, use=False, index_filt=0):
        self.use, self.index_filt = use, index_filt

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.use:
            Xfit = [D[self.index_filt] for D in X]
        else:
            Xfit = X
        return Xfit

class GIN(Model):
    def __init__(self, channels, n_layers, n_out):
        super().__init__()
        tf.random.set_seed(0)
        self.convs = []
        for l in range(n_layers-1):
            tf.random.set_seed(l)
            self.convs.append(GINConv(channels, epsilon=0, mlp_hidden=[channels, channels], kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        tf.random.set_seed(n_layers-1)
        self.conv2 = GINConv(channels, epsilon=0, mlp_hidden=[channels, n_out], kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        x, a = inputs[0], inputs[1]
        for conv in self.convs:
            x = conv([x, a])
        x = self.conv2([x, a])
        return x

dataset                   = sys.argv[1]
initial_learning_rate     = float(sys.argv[2])
batch_size                = int(sys.argv[3])
num_epochs                = int(sys.argv[4])
cv                        = int(sys.argv[5])
distance                  = sys.argv[6]
numdir                    = int(sys.argv[7])
hcard                     = int(sys.argv[8])
hdim                      = int(sys.argv[9])
fold                      = int(sys.argv[10])

use_sliced = True if distance == 'SW' else False
if use_sliced:
    thetainit = np.linspace(-np.pi/2, np.pi/2, num=numdir)
    
if (dataset[:2] == 'vs') or (dataset == 'all'):
    data_type = 'MNIST'
else:
    data_type = 'GRAPHS'

use_spektral = True

if data_type == 'MNIST':

    step = 1
    Cinit = np.array([0., np.pi/2])
    num_filts = len(Cinit)

    X = tf.keras.datasets.mnist.load_data()

    if dataset[:2] == 'vs':
        l1, l2 = int(dataset[2]), int(dataset[3])
        tridxs1, tridxs2 = np.argwhere(X[0][1] == l1).ravel()[::step], np.argwhere(X[0][1] == l2).ravel()[::step]
        teidxs1, teidxs2 = np.argwhere(X[1][1] == l1).ravel()[::step], np.argwhere(X[1][1] == l2).ravel()[::step]
        IMG = [X[0][0][j] for j in tridxs1]    + [X[0][0][j] for j in tridxs2]    + [X[1][0][j] for j in teidxs1]    + [X[1][0][j] for j in teidxs2]
        LAB = [0 for _ in range(len(tridxs1))] + [1 for _ in range(len(tridxs2))] + [0 for _ in range(len(teidxs1))] + [1 for _ in range(len(teidxs2))]
        ntrain = len(tridxs1) + len(tridxs2)
        ntot = len(tridxs1) + len(tridxs2) + len(teidxs1) + len(teidxs2)
        train_idxs, test_idxs = np.arange(0, ntrain), np.arange(ntrain, ntot)
    else:
        IMG = [X[0][0][j] for j in range(len(X[0][0]))][::step] + [X[1][0][j] for j in range(len(X[1][0]))][::step]
        LAB = [X[0][1][j] for j in range(len(X[0][0]))][::step] + [X[1][1][j] for j in range(len(X[1][0]))][::step]
        ntrain = len(np.arange(len(X[0][0]))[::step])
        train_idxs, test_idxs = np.arange(0, ntrain), np.arange(ntrain, len(IMG))

    DGMb = []
    for pdi in range(num_filts):
        DGMi = []
        for j, img in enumerate(IMG):
            inds = np.argwhere(img > 0)
            I = np.inf * np.ones(img.shape)
            for i in range(len(inds)):
                val = np.cos(Cinit[pdi])*inds[i,0] + np.sin(Cinit[pdi])*inds[i,1]
                I[inds[i,0], inds[i,1]] = val
            ccb = gd.CubicalComplex(top_dimensional_cells=I)
            ccb.persistence()
            dgmb = ccb.persistence_intervals_in_dimension(hdim)
            DGMi.append(dgmb)

            if j == -1:
                vm, vM = min(list(I[I!=np.inf].flatten())), max(list(I[I!=np.inf].flatten()))
                plt.figure()
                plt.imshow(I, vmin=vm, vmax=vM)
                plt.colorbar()
                plt.savefig(str(j) + '_' + '{:.2f}'.format(Cinit[pdi]) + '.png')

        DGMb.append(DGMi)

elif data_type == 'GRAPHS':

    path = path + 'data/graphs/' + dataset + '/mat/'
    pad_size = 50
    np.random.seed(0)

    if use_spektral:
        data = TUDataset(dataset)
        channels = 32
        layers = 3
        num_filts = 2
        gnn = GIN(channels, layers, num_filts)
         
    else:
        Cinit = np.random.uniform(size=[2,pad_size])
        num_filts = len(Cinit)

    graph_names = []

    if use_spektral:
        FUNC, LAB = [], []
        for i in range(len(data)):

            if 'x' not in data[i].keys:
                data[i].x = np.ones([data[i].a.shape[0], 10])

            LAB.append(np.argmax(data[i].y))
            
            loader = SingleLoader(data[i:i+1], epochs=1)
            for b in loader:
                FUNC.append(gnn([b[0][0], b[0][1]]).numpy())

            graph_names.append('SPK' + str(i))

    else:
        list_graph_names = np.array(os.listdir(path))    
        EVA, EVE, LAB = [], [], []
        for graph in list_graph_names:
            if graph[-4:] == '.mat':
                graph_names.append(graph)
                name = graph.split('_')
                gid = int(name[name.index('gid') + 1]) - 1
                A = np.array(loadmat(path + graph)['A'], dtype=np.float32)
                num_vertices = A.shape[0]
                label = int(name[name.index('lb') + 1])
                L = csgraph.laplacian(A, normed=True)
                egvals, egvectors = eigh(L)
                eigenvectors = np.zeros([num_vertices, pad_size])
                eigenvals = np.zeros(pad_size)
                eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
                eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]
                EVA.append(eigenvals)
                EVE.append(eigenvectors)
                LAB.append(label)

    DGMb = []
    for pdi in range(num_filts):
        DGMi = []
        for i, graph in enumerate(graph_names):

            if use_spektral:
                funcb = FUNC[i][:,pdi]
            else:
                funcb = np.sum(np.multiply(Cinit[pdi,:], EVE[i]), axis=1)

            stb = gd.SimplexTree()
            f = open(path + graph + '_st.txt', 'r')
            for line in f:
                ints = line.split(' ')
                s = [int(v) for v in ints[:-1]]
                stb.insert(s, -1e10)
            f.close()
         
            for v in range(stb.num_vertices()):
                stb.assign_filtration([v], funcb[v])
            stb.make_filtration_non_decreasing()
            stb.persistence()
            dgmb = stb.persistence_intervals_in_dimension(hdim)
            DGMi.append(dgmb)
        DGMb.append(DGMi)

num_filts = len(DGMb)
num_diags = len(DGMb[0])

for f in range(num_filts):
    dgms = np.vstack(DGMb[f]) 
    if dgms[  np.argwhere(dgms[:,1]!=np.inf).ravel()  ].shape[0] == 0:
        DGMb[f] = [np.array([[0.,0.],[1.,1.]]) for _ in range(num_diags)]

DGMb = [[DGMb[f][i] for f in range(num_filts)] for i in range(num_diags)]

if data_type == 'GRAPHS':
    kf = StratifiedKFold(n_splits=cv, shuffle=False)
    kfidx = 0
    for tr, te in kf.split(np.arange(len(LAB)), LAB):
        if kfidx == fold:
            train_idxs, test_idxs = tr, te
        kfidx += 1

train_dgmbs, test_dgmbs = [DGMb[i] for i in train_idxs], [DGMb[i] for i in test_idxs]
train_labs,  test_labs  = [LAB[i]  for i in train_idxs], [LAB[i]  for i in test_idxs]
le = LabelEncoder().fit(train_labs + test_labs)
train_labs, test_labs = le.transform(train_labs), le.transform(test_labs)

pipe = Pipeline([
    ('Feats', FeatureUnion([  ('Pipe' + str(nf), Pipeline([('Selector',  FiltrationSelector()),
                                                           ('Separator', sktda.DiagramSelector(limit=np.inf, point_type='finite')),
                                                           ('TDA',       sktda.Landscape())
                                                          ])) for nf in range(num_filts)
                           ])),
    ('Estimator', RandomForestClassifier())
])
param = {'Feats__Pipe0__Selector__use':        True,
         'Feats__Pipe0__Selector__index_filt': 0,
         'Feats__Pipe0__Separator__use':       True,
         'Feats__Pipe0__TDA__resolution':      100,
         'Feats__Pipe0__TDA__num_landscapes':  5}
for nf in range(num_filts-1):
    newparam = {'Feats__Pipe' + str(nf+1) + '__Selector__use':        True,
                'Feats__Pipe' + str(nf+1) + '__Selector__index_filt': nf+1,
                'Feats__Pipe' + str(nf+1) + '__Separator__use':       True,
                'Feats__Pipe' + str(nf+1) + '__TDA__resolution':      100,
                'Feats__Pipe' + str(nf+1) + '__TDA__num_landscapes':  5}
    param.update(newparam)
param['Estimator__random_state'] = 0

modelb = pipe.set_params(**param)
modelb.fit(train_dgmbs, train_labs)
trb = modelb.score(train_dgmbs, train_labs)
teb = modelb.score(test_dgmbs,  test_labs)

if data_type == 'MNIST':
    train_imgs, test_imgs = np.vstack([IMG[i].flatten()[np.newaxis,:] for i in train_idxs]), np.vstack([IMG[i].flatten()[np.newaxis,:] for i in test_idxs])
    rf = RandomForestClassifier().fit(train_imgs, train_labs)
    trbb = rf.score(train_imgs, train_labs)
    tebb = rf.score(test_imgs,  test_labs)
elif data_type == 'GRAPHS' and use_spektral == False:
    train_evas, test_evas = np.vstack([EVA[i].flatten()[np.newaxis,:] for i in train_idxs]), np.vstack([EVA[i].flatten()[np.newaxis,:] for i in test_idxs])
    rf = RandomForestClassifier().fit(train_evas, train_labs)
    trbb = rf.score(train_evas, train_labs)
    tebb = rf.score(test_evas,  test_labs)

if data_type == 'MNIST' or use_spektral == False:
    C = tf.Variable(initial_value=np.array(Cinit, dtype=np.float32), trainable=True)
if use_sliced:
    thetas = tf.Variable(initial_value=np.array(thetainit, dtype=np.float32), trainable=False)

#lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=initial_learning_rate, decay_steps=10, decay_rate=.01, staircase=False, name=None)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.)
#sigma = 0.001

lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1e5, decay_rate=0.99, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

batch_size = min(batch_size, len(train_idxs))

if distance == 'SW':
    filename = dataset + '_' + str(initial_learning_rate) + '_' + str(batch_size) + '_' + distance + str(numdir)
else:
    filename = dataset + '_' + str(initial_learning_rate) + '_' + str(batch_size) + '_' + distance
    
floss    = open('losses_' + filename + '_fold' + str(fold) + '.txt', 'w') 
fcoeffs  = open('coeffs_' + filename + '_fold' + str(fold) + '.txt', 'w') 
faccs    = open('accs_'   + filename + '_fold' + str(fold) + '.txt', 'w')


for epoch in range(num_epochs+1):
    
    np.random.seed(int(1e2*epoch))

    npts, total_n, batch_i = 0, len(train_idxs), 0
    perm = np.random.permutation(total_n)
    Lgrads = []

    while npts < total_n:

        batch = perm[batch_i*batch_size:(batch_i+1)*batch_size]
        npts += batch_size
        batch_i += 1
        weight = len(batch)/total_n

        #batch = np.random.choice(train_idxs, batch_size, replace=False)
        batch_labs = [LAB[i] for i in batch]
    
        with tf.GradientTape() as tape:

            dists = []
            for nf in range(num_filts):
        
                if data_type == 'MNIST':
                    dgms = []
                    for i in batch:
                        img = IMG[i]
                        inds = np.argwhere(img > 0)
                        IX, IY = 1e3 * np.ones(img.shape), 1e3 * np.ones(img.shape)
                        for k in range(len(inds)):
                            IX[inds[k,0], inds[k,1]] = inds[k,0]
                            IY[inds[k,0], inds[k,1]] = inds[k,1]
                        II = tf.math.cos(C[nf])*IX + tf.math.sin(C[nf])*IY
                        dgm = CubicalModel(II, dim=hdim, card=hcard).call()
                        dgms.append(dgm)

                elif data_type == 'GRAPHS':
                    dgms = []
                    for i in batch:
    
                        if use_spektral:
                            loader = SingleLoader(data[i:i+1], epochs=1)
                            for b in loader:
                                func = gnn([ b[0][0], b[0][1] ])[:,nf]
                        else:
                            func = tf.math.reduce_sum(tf.math.multiply(C[nf,:], tf.constant(EVE[i], dtype=tf.float32)), axis=1)

                        dgm = SimplexTreeModel(F=func, stbase=path+graph_names[i]+'_st.txt', dim=hdim, card=hcard).call()
                        dgms.append(dgm)
        
                proj_dgms = tf.linalg.matmul(tf.concat(dgms,axis=0), .5*tf.ones([2,2], tf.float32))
                dgms_big = tf.concat([tf.reshape(tf.concat([dgm, proj_dgms[:hcard*idg], proj_dgms[hcard*(idg+1):]], axis=0), [-1,2,1,1]) for idg, dgm in enumerate(dgms)], axis=2)
                cosines, sines = tf.math.cos(thetas), tf.math.sin(thetas)
                vecs = tf.concat([tf.reshape(cosines,[1,1,1,-1]), tf.reshape(sines,[1,1,1,-1])], axis=1)
                theta_projs = tf.sort(tf.math.reduce_sum(tf.math.multiply(dgms_big, vecs), axis=1), axis=0)
                t1 = tf.reshape(theta_projs, [hcard*len(batch),-1,1,numdir]) #[hcard*batch_size,-1,1,numdir])
                t2 = tf.reshape(theta_projs, [hcard*len(batch),1,-1,numdir]) #[hcard*batch_size,1,-1,numdir])
                dists.append(tf.math.reduce_mean(tf.math.reduce_sum(tf.math.abs(t1-t2), axis=0), axis=2))

            loss = 0.
            classes = np.unique(batch_labs)
            for l in classes:
                lidxs = np.argwhere(np.array(batch_labs) == l).ravel()
                idxs1 = list(itertools.product(lidxs, lidxs))
                idxs2 = list(itertools.product(lidxs, range(len(batch))))
                for nf in range(num_filts):
                    cost1 = tf.math.reduce_sum(tf.gather_nd(dists[nf], idxs1))
                    cost2 = tf.math.reduce_sum(tf.gather_nd(dists[nf], idxs2))
                    if cost2 > 0:
                        loss += cost1 / cost2
                    else:
                        loss += cost1

        if data_type == 'GRAPHS' and use_spektral:
            gradients = tape.gradient(loss, gnn.trainable_variables)
            #np.random.seed(epoch)
            #gradients[0] = tf.convert_to_tensor(gradients[0]) + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
            #optimizer.apply_gradients(zip(gradients, gnn.trainable_variables))
        else:
            gradients = tape.gradient(loss, [C]) 
            #np.random.seed(epoch)
            #gradients[0] = tf.convert_to_tensor(gradients[0]) + np.random.normal(loc=0., scale=sigma, size=gradients[0].shape)
            #optimizer.apply_gradients(zip(gradients, [C]))

        Lgrads.append(weight*gradients[0])

    final_grad = gradients
    final_grad[0] = tf.math.add_n(Lgrads)

    if data_type == 'GRAPHS' and use_spektral:
        optimizer.apply_gradients(zip(final_grad, gnn.trainable_variables))
    else:
        optimizer.apply_gradients(zip(final_grad, [C]))


    floss.write(str(loss.numpy()) + '\n')
    floss.flush()

    if data_type == 'GRAPHS' and use_spektral:
        curr_coeff = [0]
    else:
        curr_coeff = C.numpy().flatten()

    fcoeffs.write(' '.join([str(c) for c in curr_coeff]) + '\n')
    fcoeffs.flush()

    if epoch % 10 == 0:

        if data_type == 'GRAPHS' and use_spektral:
            final_coeff = [0]
        else:
            final_coeff = C.numpy()

        if data_type == 'MNIST':
            DGM = []
            for nf in range(num_filts):
                DGMi = []
                for i, img in enumerate(IMG):
                    inds = np.argwhere(img > 0)
                    I = np.inf * np.ones(img.shape)
                    for i in range(len(inds)):
                        val = np.cos(final_coeff[nf])*inds[i,0] + np.sin(final_coeff[nf])*inds[i,1]
                        I[inds[i,0], inds[i,1]] = val
                    cc = gd.CubicalComplex(top_dimensional_cells=I)
                    cc.persistence()
                    dgm = cc.persistence_intervals_in_dimension(hdim)
                    DGMi.append(dgm)
                DGM.append(DGMi)

        elif data_type == 'GRAPHS':
            DGM = []
            for nf in range(num_filts):
                DGMi = []
                for i, graph in enumerate(graph_names):

                    if use_spektral:
                        loader = SingleLoader(data[i:i+1], epochs=1)
                        for b in loader:
                            func = gnn([ b[0][0], b[0][1] ])[:,nf].numpy()
                    else:
                        func = np.sum(np.multiply(final_coeff[nf,:], EVE[i]), axis=1)

                    st = gd.SimplexTree()
                    f = open(path + graph + '_st.txt', 'r')
                    for line in f:
                        ints = line.split(' ')
                        s = [int(v) for v in ints[:-1]]
                        st.insert(s, -1e10)
                    f.close()
                    for v in range(st.num_vertices()):
                        st.assign_filtration([v], func[v])
                    st.make_filtration_non_decreasing()
                    st.persistence()
                    dgm = st.persistence_intervals_in_dimension(hdim)
                    DGMi.append(dgm)
                DGM.append(DGMi)

        for f in range(num_filts):
            dgms = np.vstack(DGM[f]) 
            if dgms[  np.argwhere(dgms[:,1]!=np.inf).ravel()  ].shape[0] == 0:
                DGM[f] = [np.array([[0.,0.],[1.,1.]]) for _ in range(num_diags)]


        num_filts = len(DGM)
        num_diags = len(DGM[0])
        DGM = [[DGM[f][i] if DGM[f][i].shape[0] != 0 else np.array([[0.,0.],[1.,1.]]) for f in range(num_filts)] for i in range(num_diags)]
        train_dgms, test_dgms = [DGM[i] for i in train_idxs], [DGM[i] for i in test_idxs]

        model = pipe.set_params(**param)
        model.fit(train_dgms, train_labs)
        tr = model.score(train_dgms, train_labs)
        te = model.score(test_dgms,  test_labs)

        if data_type == 'GRAPHS' and use_spektral:
            faccs.write(str(trb) + ' ' + str(teb) + ' ' + str(tr) + ' ' + str(te) + '\n')
        else:
            faccs.write(str(trbb) + ' ' + str(tebb) + ' ' + str(trb) + ' ' + str(teb) + ' ' + str(tr) + ' ' + str(te) + '\n')
        faccs.flush()

faccs.close()
floss.close()
fcoeffs.close()
