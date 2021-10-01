import numpy as np
import pandas as pd
import sys, os, argparse, json,time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import pickle5 as pickle
from tensorflow.keras.models import load_model

sys.path.insert(0, "../utils/")
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
parser.add_argument('--missingRate', type=str, default='0.2')
parser.add_argument('--sensor', type=str, default='acc')
parser.add_argument('--trial', type=int, default=1)
parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	inPathDataset = '/storage/datasets/sensors/LOSO/'
	args.inPath = "/storage/datasets/HAR/Reconstructed/"
	
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


def summarizeMetric(resList):
	"""
	resList: list of dictionaries
	"""
	resp = dict()
	mse = [i['MSE'] for i in resList]
	mape = [i['MAPE'] for i in resList]
	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse),scale=st.sem(mse))
	resp['MSE_down'] = icMse[0]
	resp['MSE_up'] = icMse[1]
	resp['MSE'] =  np.mean(mse)
	
	icMape = st.t.interval(alpha=0.95, df=len(mape) - 1, loc=np.mean(mape), scale=st.sem(mape))
	resp['MAPE_down'] = icMape[0]
	resp['MAPE_up'] = icMape[1]
	resp['MAPE'] =  np.mean(mape)
	
	corrX = [i['corrX'] for i in resList]
	corrY = [i['corrY'] for i in resList]
	corrZ = [i['corrZ'] for i in resList]
	resp['corrX'] = np.mean(corrX)
	resp['corrY'] = np.mean(corrY)
	resp['corrZ'] = np.mean(corrZ)
	resp['MSE_list'] = mse
	resp['corr_list'] = [np.mean([a,b,c]) for a,b,c in zip(corrX,corrY, corrZ)]
	
	return resp


def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return m, ic


if __name__ == '__main__':
	# process the data:
	metricsAEy = []
	classifResult = {}
	classifResult['AEy_acu'] = []
	classifResult['AEy_rec'] = []
	classifResult['AEy_f1'] =[]
	start = time.time()
	if args.sensor == 'acc':
		s_idx = 0
		sensor_factor = '1.0'
		sensor = args.sensor
	elif args.sensor == 'gyr':
		s_idx = 0
		sensor_factor = '0.1'
		sensor = args.sensor
	elif args.sensor == "accGyr":
		s_idx = slice(0,2)
		sensor_factor = '1.1'

		mR = str(int(float(args.missingRate) * 100))
		recFile = os.path.join(args.inPath, 'acc', f'USCHAD_recAEyacc_miss{mR}_{args.trial}.npz')
		with np.load(recFile, allow_pickle=True) as tmp:
			XAcc = tmp['X']
			idxAcc = tmp['idx']
		sensor = 'gyr'
		
	

	mR = str(int(float(args.missingRate) * 100))
	recFile = os.path.join(args.inPath, sensor, f'USCHAD_recAEy{sensor}_miss{mR}_{args.trial}.npz')
	with np.load(recFile, allow_pickle=True) as tmp:
		X = tmp['X']
		idx = tmp['idx']
		#yTrue = tmp["y_list"]
	
	for fold_i in range(args.Nfolds):
		
		# Getting the original data and the missing data version
		# inPath = '/storage/datasets/sensors/LOSO/'
		fileName = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{fold_i}'
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor=sensor_factor, path=inPathDataset)
		DH.splitTrainTest(fold_i=fold_i)
		testSensor = DH.dataXtest[s_idx]
		yTrue = DH.dataYtest
		
		Xrec = X[fold_i]
		if args.sensor =='accGyr':
			Xrec = np.concatenate([Xrec,XAcc[fold_i]],axis = 2)
		
		

		model = load_model(os.path.join(classifiersPath, f'DCNN_{args.sensor}_USCHAD_fold_{fold_i}.h5'))

		# classificacao com os dados reconstruidos do AutoENcoder:
		pred = model.predict(np.expand_dims(Xrec, axis=-1))
		pred = np.argmax(pred,axis = 1)
		classifResult['AEy_acu'].append(accuracy_score(yTrue, pred))
		classifResult['AEy_rec'].append(recall_score(yTrue, pred, average='macro'))
		classifResult['AEy_f1'].append(f1_score(yTrue, pred, average='macro'))
		am = absoluteMetrics(testSensor,Xrec)
		res = am.runAll()
		metricsAEy.append(res)
		del am
		del DH
		

	metricsAEy = summarizeMetric(metricsAEy)
	print('missing Rate:', args.missingRate, '\n')
	print('Sensor: ', args.sensor,'   ')
	print('Trial:', args.trial, '\n')
	print('AEy: \n',metricsAEy)
	print('\n\n')
	resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))
	print(resultClass)
	print('\n\n')
	end = time.time()
	print('time:  ',(end - start)/60)
