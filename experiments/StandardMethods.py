import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import os
import sys
import json
import pickle5 as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate',type=str,default= '0.5')
parser.add_argument('--Nfolds',type=int,default= 14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	sys.path.insert(0, "/home/guilherme.silva/classifiers")
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")
	from Catal import Catal
else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\"
	sys.path.insert(0, args.outPath )
	from Catal import Catal
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath =  os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\trained\\")

if __name__ == '__main__':
	# process the data:
	metrics = []
	AM = absoluteMetrics()
	for fold_i in range(args.Nfolds):
		fileName = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{fold_i}'
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor='1.1', path=args.inPath)
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor='1.0')
		DH.splitTrainTest(fold_i=fold_i)
		testMiss = np.concatenate(DH.dataXmissingTest, axis=-1)
		test = np.concatenate(DH.dataXtest, axis=-1)
		y = DH.dataYtest
		
		with open(os.path.join(classifiersPath, f'Catal_USCHAD_{fold_i}.pkl'),'rb') as inp:
			catal_classifier = pickle.load(inp)
		sm = SM()
		xRec = sm.runAll(testMiss)
		del sm
		yPred = catal_classifier.predict(xRec)
		mse = AM.myMSE(test, xRec)
		acc = accuracy_score(yPred, y)
		f1 = f1_score(yPred, y, average='macro')
		metrics.append([mse, acc, f1])
		del DH
	
	metricsM = np.mean(metrics, axis=0)
	print(metricsM)
	result = {}
	result['MSE'] = str(metricsM[0])
	result['Acuracy'] = str(metricsM[1])
	result['f1'] = str(metricsM[2])
	savePath = os.path.join(args.outPath, f'result_{args.method}_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(result, write_file)
	np.save(savePath + 'ALL', metrics=metrics)