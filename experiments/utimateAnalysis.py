import numpy as np
import pandas as pd
import sys,os, argparse,json,time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import pickle5 as pickle
from tensorflow.keras.models import load_model

sys.path.insert(0,"../utils/")
from dataHandler import dataHandler
from metrics import absoluteMetrics

"""
This experiment aims to mensure the quality of accelerometer reconstruction.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate', type=str, default='0.5')
parser.add_argument('--sensor', type=str, default='accGyr')
parser.add_argument('--method', type=str, default='matrixFactorization')
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	inPathDataset = '/storage/datasets/sensors/LOSO/'
	args.inPath ="/storage/datasets/HAR/Reconstructed/"
	
	args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")

	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\Reconstructed\\'
	inPathDataset = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\results\\"
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\")
	from plotGenerator import plot_result


def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return m,ic
if __name__ == '__main__':
	# process the data:
	metrics =[]
	classifResult = {}
	classifResult[args.method + '_acc'] = []
	classifResult[args.method + '_rec'] = []
	classifResult[args.method + '_f1'] =[]
	classifResult['acc_f1'] = []
	classifResult['gyr_f1'] = []
	
		
	if args.sensor =='acc':
		missing_sensor = '1.0'
		s_idx = 0
		ini = 0
		end = 3

	elif args.sensor =='gyr':
		missing_sensor = '0.1'
		s_idx = 1
		ini = 3
		end = 6

	elif args.sensor =='accGyr':
		missing_sensor = '0.1'
		s_idx = slice(0,2)
		ini = 0
		end = 6

	start = time.time()
	print('starting')
	for fold_i in range(args.Nfolds):
		#Getting the original data and the missing data version
		#inPath = '/storage/datasets/sensors/LOSO/'
		fileName = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{fold_i}'
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor='1.1', path=inPathDataset)
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor=missing_sensor)
		DH.splitTrainTest(fold_i=fold_i)
		
		testSensor = DH.dataXtest[s_idx]
		yTrue = DH.dataYtest

		model = load_model(os.path.join(classifiersPath, f'DCNN_{args.sensor}_USCHAD_fold_{fold_i}.h5'))


		testMiss = np.concatenate(DH.dataXmissingTest, axis=-1)
		idxMissTest = DH.missing_indices['test']
		del DH
		#Reconstruction with standard methods
		sm = SM()
		works, xRec = sm.runMethod(testMiss, args.method)
		del testMiss
		del sm
		if args.sensor =='accGyr':
			modelAcc = load_model(os.path.join(classifiersPath, f'DCNN_acc_USCHAD_fold_{fold_i}.h5'))
			modelGyr = load_model(os.path.join(classifiersPath, f'DCNN_gyr_USCHAD_fold_{fold_i}.h5'))
			DH = dataHandler()
			DH.load_data(dataset_name=args.dataset, sensor_factor='1.1', path=inPathDataset)
			DH.apply_missing(missing_factor=args.missingRate, missing_sensor='1.0')
			DH.splitTrainTest(fold_i=fold_i)
			testMiss = np.concatenate(DH.dataXmissingTest, axis=-1)
			sm = SM()
			works, xRecAcc = sm.runMethod(testMiss, args.method)
			del testMiss
			del sm
			del DH
			
			yPredAcc = modelAcc.predict(np.expand_dims(xRecAcc[:,:,0:3], axis=-1))
			yPredAcc = np.argmax(yPredAcc, axis=1)
			
			yPredGyr = modelGyr.predict(np.expand_dims(xRec[:,:,3:6], axis=-1))
			yPredGyr = np.argmax(yPredGyr, axis=1)
			
			xRecF = np.concatenate([xRecAcc[:,:,0:3], xRec[:,:,3:6]], axis=2)
			classifResult['acc_f1'].append(f1_score(yTrue, yPredAcc, average='macro'))
			classifResult['gyr_f1'].append(f1_score(yTrue, yPredGyr, average='macro'))
			
			
			testSensor = np.concatenate(testSensor,axis=2)
		else:
			xRecF = xRec

		# run the evaluating metrics and classification
		# am = absoluteMetrics(testSensor, xRec[:, :, ini:end], idxMissTest)
		# res = am.runAll()
		# metrics.append(res)
		#del am
		yPred = model.predict(np.expand_dims(xRecF[:, :, ini:end], axis = -1))
		yPred = np.argmax(yPred, axis=1)
		

		classifResult[args.method  + '_acc'].append(accuracy_score(yTrue, yPred))
		classifResult[args.method + '_rec'].append(recall_score(yTrue, yPred, average='macro'))
		classifResult[args.method + '_f1'].append(f1_score(yTrue, yPred, average='macro'))

	#
	#metrics = absoluteMetrics.summarizeMetric(metrics)


	print('missing Rate:',args.missingRate,'\n')
	print('Sensor:', args.sensor, '\n')
	print('Trial:', args.trial, '\n')
	print('Method: ',args.method)
	#print(args.method,'\n',metrics)

	resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))
	print(resultClass)
	print('\n')
	end = time.time()
	print('time:  ', (end - start) / 60)
