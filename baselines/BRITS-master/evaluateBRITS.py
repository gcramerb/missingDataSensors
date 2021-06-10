import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import os
import sys
import json
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--dataset', type=str, default="USCHAD")

args = parser.parse_args()

if (args.inPath is not None):
	sys.path.insert(0, "/home/guilherme.silva/missing_data")
	from dataHandler import dataHandler
	sys.path.insert(0, "/home/guilherme.silva/classifiers")
	from Catal import Catal
else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\"
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	from dataHandler import dataHandler
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\")
	from Catal import Catal

def trainClassifiers(datasetName,inPath,outPath):
	# dataPreparation:
	tmp = np.load(os.path.join(inPath, f'{datasetName}.npz'), allow_pickle=True)
	X = tmp['X']
	X = X[:, 0, :, :]
	y = tmp['y']
	folds = tmp['folds']
	n_class = y.shape[1]
	y = np.argmax(y, axis=1)
	# - ------------------------
	
	for i in range(0, len(folds)):
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		X_train = X[train_idx]
		# concatenate gyr + acc_reconstructed:
		catal_classifier = Catal()
		catal_classifier.fit(X_train, y[train_idx])
		with open(f'Catal_{datasetName}_{i}.pkl', 'wb') as output:
			pickle.dump(catal_classifier, outPath, pickle.HIGHEST_PROTOCOL)
		del catal_classifier
		
def classificationResult(data,Fold,path):
	with open(os.path.join(path,f'Catal_USCHAD_{Fold}.pkl'), 'rb') as input:
		catal_classifier = pickle.load(input)
	yPred = catalClassifier.predict(data)
	return yPred
	
	
if __name__ == '__main__':
	# Paper: On the use of ensemble of classifiers for accelerometer-based activity recognition
	np.random.seed(12227)
	trainClassifiers(args.dataset,args.inPath,args.outPath)
	

	
	

	
