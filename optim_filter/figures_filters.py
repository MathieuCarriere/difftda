import matplotlib.pyplot as plt
import numpy as np
import sys
import os

dataset = sys.argv[1]
folds   = int(sys.argv[2])

list_files = os.listdir('./')
faccsall = [f for f in list_files if f[:4] == 'accs']
flossesall = [f for f in list_files if f[:4] == 'loss']
fcoeffsall = [f for f in list_files if f[:4] == 'coef']
faccs = [f for f in faccsall if f.split('_')[1] == dataset]
flosses = [f for f in flossesall if f.split('_')[1] == dataset]
fcoeffs = [f for f in fcoeffsall if f.split('_')[1] == dataset]

if not folds:

	try:
		accs = np.loadtxt(faccs[0], dtype=np.float32)
	except ValueError:
		f = open(faccs[0], 'r')
		L = f.readlines()[:-1]
		accs = np.array([[float(a) for a in l[:-1].split(' ')[:-1]] for l in L])

	losses = np.loadtxt(flosses[0], dtype=np.float32)	
	plt.figure()
	plt.plot(losses)
	plt.savefig('Losses_' + dataset + '.png')

	coeffs = np.loadtxt(fcoeffs[0], dtype=np.float32)
	plt.figure()
	if len(coeffs.shape) == 2:
		for c in range(coeffs.shape[1]):
			plt.plot(coeffs[:,c])
		plt.savefig('Coeffs_' + dataset + '.png')
	else:
		plt.plot(coeffs)
		plt.savefig('Coeffs_' + dataset + '.png')

else:

	AS, LS = [], []
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

	for f in flosses:
		i = f[-5]
		LS.append((i, np.loadtxt(f, dtype=np.float32)))
	plt.figure()
	for loss in LS:
		plt.plot(loss[1], label='Fold ' + loss[0])
	plt.legend()
	plt.savefig('Losses_' + dataset + '.png')

	for f in fcoeffs:
		i = f[-5]
		CS = np.loadtxt(f, dtype=np.float32)
		plt.figure()
		if len(CS.shape) == 2:
			for c in range(CS.shape[1]):
				plt.plot(CS[:,c])
			plt.savefig('Coeffs_' + dataset + '_fold' + i + '.png')
		else:
			plt.plot(CS)
			plt.savefig('Coeffs_' + dataset + '_fold' + i + '.png')


	
#TRBB, TEBB, TRB, TEB, TR, TE = accs[:,0::6], accs[:,1::6], accs[:,2::6], accs[:,3::6], accs[:,4::6], accs[:,5::6]
TRB, TEB, TR, TE = accs[:,0::4], accs[:,1::4], accs[:,2::4], accs[:,3::4]

#trbb, std_trbb = np.mean(TRBB, axis=1), np.std(TRBB, axis=1)
#tebb, std_tebb = np.mean(TEBB, axis=1), np.std(TEBB, axis=1)
trb, std_trb = np.mean(TRB, axis=1), np.std(TRB, axis=1)
teb, std_teb = np.mean(TEB, axis=1), np.std(TEB, axis=1)
tr, std_tr  = np.mean(TR, axis=1), np.std(TR, axis=1)
te, std_te  = np.mean(TE, axis=1), np.std(TE, axis=1)

print(trb, std_trb, teb, std_teb, tr, std_tr, te, std_te)

plt.figure()

#plt.plot(trbb, label='w/o topo tr', c='red', linestyle='dashed')
#plt.fill_between(range(len(TRBB)), trbb-std_trbb, trbb+std_trbb, color='red', alpha=.1)
plt.plot(trb, label='non optim. topo tr', c='green', linestyle='dashed')
plt.fill_between(range(len(TRB)), trb-std_trb, trb+std_trb, color='green', alpha=.1)
plt.plot(tr, label='optim. topo tr', c='blue', linestyle='dashed')
plt.fill_between(range(len(TR)), tr-std_tr, tr+std_tr, color='blue', alpha=.1)

#plt.plot(tebb, label='w/o topo te', c='red')
#plt.fill_between(range(len(TEBB)), tebb-std_tebb, tebb+std_tebb, color='red', alpha=.1)
plt.plot(teb, label='non optim. topo te', c='green')
plt.fill_between(range(len(TEB)), teb-std_teb, teb+std_teb, color='green', alpha=.1)
plt.plot(te, label='optim. topo te', c='blue')
plt.fill_between(range(len(TE)), te-std_te, te+std_te, color='blue', alpha=.1)

plt.legend()
plt.savefig('Accs_' + dataset + '.png')
