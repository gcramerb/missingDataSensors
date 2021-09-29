import numpy as np
import pandas as pd
import sys,os, argparse,json
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
parser.add_argument('--missingRate', type=str, default='0.2')
parser.add_argument('--sensor', type=str, default='acc')
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

def summarizeMetric(resList):
	resp = dict()
	mse = [i['MSE'] for i in resList]
	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse),scale=st.sem(mse))
	mse = np.mean(mse)
	resp['MSE'] =  np.mean(mse)
	resp['MSE_down'] = icMse[0]
	resp['MSE_up'] = icMse[1]
	corrX = [i['corrX'] for i in resList]
	corrY = [i['corrY'] for i in resList]
	corrZ = [i['corrZ'] for i in resList]
	resp['corrX'] = np.mean(corrX)
	resp['corrY'] = np.mean(corrY)
	resp['corrZ'] = np.mean(corrZ)
	resp['MSE_list'] = mse
	resp['corr_list'] = [np.mean(a,b,c) for a,b,c in zip(corrX,corrY, corrZ)]

	return resp
def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return m,ic
if __name__ == '__main__':
	# process the data:
	metricsMICE = []
	metricsMF = []
	metricsEM = []
	#metricsAEy = []
	classifResult = {}
	for n_ in ['MICE','MF','EM']:
		classifResult[n_ + '_acc'] = []
		classifResult[n_ + '_rec'] = []
		classifResult[n_ + '_f1'] =[]
	
	# classifResult['AEy_acc'] = []
	# classifResult['AEy_rec'] = []
	# classifResult['AEy_f1'] =[]
	# resOri = []
		
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

		#Reconstruction with standard methods
		sm = SM()
		works, xRecMICE = sm.runMethod(testMiss,'MICE')
		works, xRecMF = sm.runMethod(testMiss, 'matrixFactorization')
		works, xRecEM = sm.runMethod(testMiss, 'expectationMaximization')
		del sm

		# run the evaluating metrics and classification
		am = absoluteMetrics(testSensor,xRecMICE[:,:,ini:end],idxMissTest)
		res = am.runAll()
		metricsMICE.append(res)
		yPredMICE = model.predict(np.expand_dims(xRecMICE[:,:,ini:end],axis = -1))
		yPredMICE = np.argmax(yPredMICE, axis=1)
		del am
		
		am = absoluteMetrics(testSensor,xRecMF[:,:,ini:end],idxMissTest)
		res = am.runAll()
		metricsMF.append(res)
		yPredMF = model.predict(np.expand_dims(xRecMF[:,:,ini:end],axis = -1))
		yPredMF = np.argmax(yPredMF, axis=1)
		del am
		
		am = absoluteMetrics(testSensor,xRecEM[:,:,ini:end],idxMissTest)
		res = am.runAll()
		metricsEM.append(res)
		yPredEM = model.predict(np.expand_dims(xRecEM[:,:,ini:end],axis = -1))
		yPredEM = np.argmax(yPredEM, axis=1)

		del am
		del DH

		for name,pred in zip(['MICE','MF','EM'],[yPredMICE,yPredMF,yPredEM]):
			classifResult[name + '_acc'].append(accuracy_score(yTrue, pred))
			classifResult[name + '_rec'].append(recall_score(yTrue, pred, average='macro'))
			classifResult[name + '_f1'].append(f1_score(yTrue, pred, average='macro'))
	#
	metricsMICE = summarizeMetric(metricsMICE)
	metricsEM = summarizeMetric(metricsEM)
	metricsMF = summarizeMetric(metricsMF)
	#metricsAEy = summarizeMetric(metricsAEy)
	print('missing Rate:',args.missingRate,'\n')
	print('Sensor:', args.sensor, '\n')
	print('Trial:', args.trial, '\n')
	print('MICE: \n\n',metricsMICE)
	print('MF: \n\n',metricsMF)
	print('EM: \n\n', metricsEM)
	# print('Aey: \n\n',metricsAEy)
	# print('\n\n\n\n')
	resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))
	print(resultClass)
	print('\n\n\n')
	# savePath = os.path.join(args.outPath, f'result_MICE_{args.dataset.split(".")[0]}_{args.missingRate}')
	# with open(savePath + '.json', "w") as write_file:
	# 	json.dump(metricsMICE, write_file)
	# savePath = os.path.join(args.outPath, f'result_EM_{args.dataset.split(".")[0]}_{args.missingRate}')
	# with open(savePath + '.json', "w") as write_file:
	# 	json.dump(metricsEM, write_file)
	# savePath = os.path.join(args.outPath, f'result_MF_{args.dataset.split(".")[0]}_{args.missingRate}')
	# with open(savePath + '.json', "w") as write_file:
	# 	json.dump(metricsMF, write_file)
	# savePath = os.path.join(args.outPath, f'result_Classification_{args.dataset.split(".")[0]}_{args.missingRate}')
	# with open(savePath + '.json', "w") as write_file:
	# 	json.dump(resultClass, write_file)