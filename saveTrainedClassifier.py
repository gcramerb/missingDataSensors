import numpy as np
import json,os,sys
import argparse,pickle
sys.path.insert(0, "/")
from utils.dataHandler import dataHandler


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--dataset', type=str, default="USCHAD")
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
	with np.load(os.path.join(args.inPath, f'{args.dataset}.npz'), allow_pickle=True) as tmp:
		X = tmp['X']
		y = tmp['y']
		folds = tmp['folds']
	X = X[:, :, :, 0:3]
	X = np.transpose(X, (0, 2, 3, 1))
	n_class = y.shape[1]
	#y = np.argmax(y, axis=1)
	for i in range(0, len(folds)):
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		clf = classifier()
		clf.fit(X[train_idx], y[train_idx])
		saveFile = os.path.join(args.outPath, f'DCNN_acc_{args.dataset}_fold_{i}.h5')
		clf.save(saveFile)
		yTrue= np.argmax(y[test_idx], axis=1)
		acc,rec,f1 = clf.metrics(X[test_idx], yTrue)
		print('\n\n\n',acc,' ',f1,'\n\n\n')
		del clf

if __name__ == '__main__':
	trainSaveClassifiers()