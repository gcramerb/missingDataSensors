import numpy as np
import pandas as pd
import sys, os, argparse, json, time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import pickle5 as pickle
from tensorflow.keras.models import load_model

sys.path.insert(0, "../utils/")
from dataHandler import dataHandler
from metrics import absoluteMetrics

"""
This experiment aims to mensure the quality of The imputation of our Methodology.
We analyse both clssification and absolute metrics.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)

parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	inPathDataset = '/storage/datasets/sensors/LOSO/'
	args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")
	
	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
else:
	inPathDataset = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\results\\"
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\")
	from plotGenerator import plot_result

from utils.dataHandler import dataHandler
def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return m, ic


if __name__ == '__main__':

	classifResult = {}
	sensors = ['acc','gyr']
	metrics = ['accuracy','f1']
	for s in sensors:
		for m in metrics:
			classifResult[s + '_' + m] = []


	for s_idx,sen in enumerate(sensors):
		for fold_i in range(args.Nfolds):
			
			# Getting the original data and the missing data version
			DH = dataHandler()
			DH.load_data(dataset_name=args.dataset, sensor_factor='1.1.0', path=inPathDataset)
			DH.splitTrainTest(fold_i=fold_i)
			testSensor = DH.dataXtest[s_idx]
			yTrue = DH.dataYtest

			model = load_model(os.path.join(classifiersPath, f'DCNN_{sen}_USCHAD_fold_{fold_i}.h5'))

			yPred = model.predict(np.expand_dims(testSensor, axis=-1))
			yPred = np.argmax(yPred, axis=1)

			classifResult[f'{sen}_accuracy'].append(accuracy_score(yTrue, yPred))
			classifResult[f'{sen}_f1'].append(f1_score(yTrue, yPred, average='macro'))

		del DH
		print('Sensor: ',sen, '   ')
		print('\n\n')
		resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))
		print(resultClass)
		print('\n\n')

