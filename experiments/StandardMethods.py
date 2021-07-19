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
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate',type=str,default= '0.9')
parser.add_argument('--Nfolds',type=int,default= 14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
parser.add_argument('--method', type=str, default="MICE")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	sys.path.insert(0, "/home/guilherme.silva/classifiers")
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")
	from Catal import Catal
	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
		
else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\"
	sys.path.insert(0, args.outPath )
	from Catal import Catal
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath =  os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\")

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
		trainX = np.concatenate(DH.dataXtrain, axis=-1)
		trainY =  DH.dataYtrain
		y = DH.dataYtest
		sm = SM()
		works,xRec = sm.runMethod(testMiss,args.method)
		if not works:
			print(f'\n\n Erro in folf{fold_i} do metodos {args.method}')
		del sm
		catal_classifier = Catal()
		catal_classifier.fit(trainX,trainY)
		yPred = catal_classifier.predict(xRec)
		mse = AM.myMSE(test, xRec)
		acc = accuracy_score(y,yPred)
		f1 = f1_score(y,yPred, average='macro')
		metrics.append([mse, acc, f1])
		del DH
	
	metricsM = np.mean(metrics, axis=0)
	metrics = np.array(metrics)
	ic_acc = st.t.interval(alpha=0.95, df=len(metrics[:,1]) - 1, loc=np.mean(metrics[:,1]), scale=st.sem(metrics[:,1]))
	ic_f1 = st.t.interval(alpha=0.95, df=len(metrics[:, 2]) - 1, loc=np.mean(metrics[:, 2]),scale=st.sem(metrics[:,2]))
	result = {}
	result['MSE'] = str(metricsM[0])
	result['Acuracy'] = str(metricsM[1])
	result['Acc_icLow'] = ic_acc[0]
	result['Acc_icHigh'] = ic_acc[1]
	result['f1'] = str(metricsM[2])
	result['F1_icLow'] = ic_f1[0]
	result['F1_icHigh'] = ic_f1[1]
	savePath = os.path.join(args.outPath, f'result_{args.method}_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(result, write_file)
	#np.save(savePath + 'ALL.npz', metrics=metrics,ic_acc = ic_acc,ic_f1 = ic_f1)