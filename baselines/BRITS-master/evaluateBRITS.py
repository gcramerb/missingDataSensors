import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
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
		with open(os.path.join(outPath,f'Catal_{datasetName}_{i}.pkl'), 'wb') as output:
			pickle.dump(catal_classifier, output, pickle.HIGHEST_PROTOCOL)
		del catal_classifier
		
def predict(data,Fold,path):
	with open(os.path.join(path,f'Catal_USCHAD_{Fold}.pkl'), 'rb') as input:
		catal_classifier = pickle.load(input)	#
	yPred = catalClassifier.predict(data)
	return yPred
	
	
if __name__ == '__main__':
	# Paper: On the use of ensemble of classifiers for accelerometer-based activity recognition
	np.random.seed(12227)
	#trainClassifiers(args.dataset,args.inPath,args.outPath)


	
	dataset = 'USCHAD.npz'
	missing_list = ['0.5']
	# missing_list = ['0.2']
	finalResult = dict()
	finalResult['acc'] = dict()
	finalResult['f1'] = dict()
	finalResult['rec'] = dict()
	for miss in missing_list:
		finalResult['acc'][miss] = list()
		finalResult['f1'][miss] = list()
		finalResult['rec'][miss] = list()
	
	# data_input_file = path + dataset
	data_input_file = os.path.abspath('C:\\Users\\gcram\\Documents\\Datasets\\USCHAD_forBRITS\\')
	classifier = "Catal"
	
	simple_impute = False
	# imputeList = ['mean']
	# imputeList = ['AEY_mse']
	
	# dataPreparation:
	tmp = np.load(os.path.join(data_input_file, 'brits_data.npy'), allow_pickle=True)
	X = tmp['X']
	X = X[:, 0, :, :]
	y = tmp['y']
	folds = tmp['folds']
	n_class = y.shape[1]
	
	avg_acc = []
	avg_recall = []
	avg_f1 = []
	y = np.argmax(y, axis=1)
	# - ------------------------
	# Result :
	acc = dict()
	f1 = dict()
	recall = dict()
	
	for i in range(0, len(folds)):
		xRec_list = []
		yRec_list = []
		
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		X_train = X[train_idx]
		
		# concatenate gyr + acc_reconstructed:
		gyr = X[test_idx][:, :, 3:6]
		for miss in missing_list:
			if simple_impute:
				DH = dataHandler()
				DH.load_data(dataset_name=dataset, sensor_factor='1.0', path=path)
				DH.apply_missing(missing_factor=miss, missing_sensor='1.0')
				DH.impute('mean')
				DH.splitTrainTest(fold_i=i)
				testRec = DH.dataXreconstructedTest[0]
				xRec = np.concatenate([testRec, gyr], axis=-1)
				yRec = DH.dataYtest
				yRec_list.append(yRec)
				xRec_list.append(xRec)
				del DH
			else:
				file = 'USCHAD' + '_' + miss + '_AEY_mse' + str(i) + '.npz'
				data = np.load(path + file, allow_pickle=True)
				testRec = data['deploy_data']
				# test = data['data']
				yRec_list.append(data['classes'])
				xRec = np.concatenate([testRec, gyr], axis=-1)
				xRec_list.append(xRec)


	
	

	
	

	
