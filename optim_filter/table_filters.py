import os
import numpy as np
import matplotlib.pyplot as plt

mnist_ds = [] #['all'] + ['vs' + str(i) + str(j) for i in range(0,9) for j in range(i+1,10)]
graphs_ds = ['PROTEINS', 'MUTAG', 'COX2', 'DHFR', 'BZR', 'FRANKENSTEIN', 'IMDB-MULTI', 'IMDB-BINARY', 'NCI1', 'NCI109']

matrix = np.zeros([10,10])

ls_datasets = mnist_ds + graphs_ds
ls_folds = [] + [1 for _ in range(len(graphs_ds))]  
           #[0 for _ in range(len(mnist_ds))]   
ls_epochs = [] + [19 for _ in range(len(graphs_ds))]
            #[50 for _ in range(len(mnist_ds))] 

with open('table.tex', 'w') as ftable:


	ftable.write('\\centering\n')
	ftable.write('\\begin{tabular}{c||c||ccc}\n')
	ftable.write('Dataset & Baseline & Before & After & Difference \\\\ \n')
	ftable.write('\\hline\n')
	list_files = os.listdir('./')

	for i, ds in enumerate(ls_datasets):

		ep = ls_epochs[i]
		folds = ls_folds[i]
		faccsall = [f for f in list_files if f[:4] == 'accs']
		faccs = [f for f in faccsall if f.split('_')[1] == ds]

		if not folds:

			try:
				accs = np.loadtxt(faccs[0], dtype=np.float32)
			except ValueError:
				f = open(faccs[0], 'r')
				L = f.readlines()[:-1]
				accs = np.array([[float(a) for a in l[:-1].split(' ')[:-1]] for l in L])

		else:

			AS = []
			for f in faccs:
				i = f[-5]
				try:
					AS.append(np.loadtxt(f, dtype=np.float32))
				except ValueError:
					F = open(f, 'r')
					L = F.readlines()[:-1]
					AS.append(np.array([[float(a) for a in l[:-1].split(' ')[:-1]] for l in L]))
			minsize = min([len(a) for a in AS])
			AS = [a[:minsize] for a in AS]
			accs = np.hstack(AS)	

		accs = 100 * accs
		#TRBB, TEBB, TRB, TEB, TR, TE = accs[:,0::6], accs[:,1::6], accs[:,2::6], accs[:,3::6], accs[:,4::6], accs[:,5::6]
		TRB, TEB, TR, TE = accs[:,0::4], accs[:,1::4], accs[:,2::4], accs[:,3::4]

		trb, std_trb = np.mean(TRB, axis=1), np.std(TRB, axis=1)
		teb, std_teb = np.mean(TEB, axis=1), np.std(TEB, axis=1)
		trbb, std_trbb = trb, std_trb #np.mean(TRBB, axis=1), np.std(TRBB, axis=1)
		tebb, std_tebb = teb, std_teb #np.mean(TEBB, axis=1), np.std(TEBB, axis=1)
		tr, std_tr  = np.mean(TR, axis=1), np.std(TR, axis=1)
		te, std_te  = np.mean(TE, axis=1), np.std(TE, axis=1)

		ep = min(ep, len(te)-1)
		if (np.abs(tebb[ep]-te[ep]) <= 1e10) or (ds in graphs_ds):
			if (ds in graphs_ds):
				if te[ep]-teb[ep] > 0.:
					ftable.write('\\texttt{' + ds + '} & ' + '{:.1f}'.format(tebb[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_tebb[ep]) + ' & ' + '{:.1f}'.format(teb[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_teb[ep]) + ' & ' + '{:.1f}'.format(te[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_te[ep]) + ' & \\bf{' + '+{:.1f}'.format(te[ep]-teb[ep]) + '} \\\\ \n')
				else:
					ftable.write('\\texttt{' + ds + '} & ' + '{:.1f}'.format(tebb[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_tebb[ep]) + ' & ' + '{:.1f}'.format(teb[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_teb[ep]) + ' & ' + '{:.1f}'.format(te[ep]) + ' $\\pm$ ' + '{:.2f}'.format(std_te[ep]) + ' & ' + '{:.1f}'.format(te[ep]-teb[ep]) + ' \\\\ \n ')
			else:
				if te[ep]-teb[ep] > 0.:
					ftable.write('\\texttt{' + ds + '} & ' + '{:.1f}'.format(tebb[ep]) + ' & ' + '{:.1f}'.format(teb[ep]) + ' & ' + '{:.1f}'.format(te[ep]) + ' & \\bf{' + '+{:.1f}'.format(te[ep]-teb[ep]) + '} \\\\ \n')
				else:
					ftable.write('\\texttt{' + ds + '} & ' + '{:.1f}'.format(tebb[ep]) + ' & ' + '{:.1f}'.format(teb[ep]) + ' & ' + '{:.1f}'.format(te[ep]) + ' & ' + '{:.1f}'.format(te[ep]-teb[ep]) + ' \\\\ \n ')


		if ds[:2] == 'vs':
			i1, i2 = int(ds[2]), int(ds[3])
			matrix[i1,i2] = te[ep]-teb[ep]
			matrix[i2,i1] = matrix[i1,i2]

	ftable.write('\\end{tabular}\n')



#fig, ax = plt.subplots(1,1)
#plt.imshow(np.flip(matrix,0), cmap='bwr', vmin=-40, vmax=40)
#plt.colorbar()
#y_label_list = np.arange(0,10)[::-1]
#ax.set_yticks(np.arange(0,10))
#ax.set_yticklabels(y_label_list)
#x_label_list = range(0,10)
#ax.set_xticks(range(0,10))
#ax.set_xticklabels(x_label_list)
#fig.savefig('improv.png')
