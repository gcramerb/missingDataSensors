import numpy as np
import json,os,sys


sys.path.insert(0, "/")
from dataHandler import dataHandler
sys.path.insert(0, "/")
from utils.dataHandler import dataHandler


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()
if args.slurm:

	args.inPath  = '/storage/datasets/sensors/LOSO/'
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/")
	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	classifiersPath = "C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\"
	
args.outPath = os.path.join(classifiersPath,'trained')
sys.path.insert(0, classifiersPath)
from DCNNclassifier import DCNNclassifier as classifier

def trainSaveClassifiers():
	# dataPreparation:
	tmp = np.load(os.path.join(args.inPath, f'{args.dataset}.npz'), allow_pickle=True)
	X = tmp['X']
	X = X[:, :, :, 0:3]
	y = tmp['y']
	folds = tmp['folds']
	n_class = y.shape[1]
	y = np.argmax(y, axis=1)
	for i in range(0, len(folds)):
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		Xtrain = X[train_idx]
		model = classifier()
		model.fit(Xtrain, y[train_idx])
		with open(os.path.join(args.outPath, f'DCNN_acc_{args.dataset}_fold_{i}.pkl'), 'wb') as output:
			pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
		del model
