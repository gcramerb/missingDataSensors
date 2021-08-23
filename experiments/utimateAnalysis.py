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
parser.add_argument('--Nfolds', type=int, default=1)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	args.inPath = '/storage/datasets/sensors/LOSO/'
	args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")

	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\results\\"
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	from utils.dataHandler import dataHandler
	from utils.metrics import absoluteMetrics
	from baselines.timeSeriesReconstruction import StandardMethods as SM
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\")

def summarizeMetric(resList):
	resp = dict()
	mse = [i['MSE'] for i in resList]
	icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse),scale=st.sem(mse))
	mse = np.mean(mse)
	resp['MSE'] = mse
	resp['MSE_down'] = icMse[0]
	resp['MSE_up'] = icMse[1]
	resp['PSNR'] = np.mean([i['PSNR'] for i in resList])
	resp['corrX'] = np.mean([i['corrX'] for i in resList])
	resp['corrY'] = np.mean([i['corrY'] for i in resList])
	resp['corrZ'] = np.mean([i['corrZ'] for i in resList])
	return resp
def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
if __name__ == '__main__':
	# process the data:
	metricsMICE = []
	metricsMF = []
	metricsEM = []
	classifResult = {}
	for name in ['MICE','MF','EM','Test']:
		classifResult[name + '_acc'] = []
		classifResult[name + '_rec'] = []
		classifResult[name + '_f1'] =[]
		

	for fold_i in range(args.Nfolds):
		fileName = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{fold_i}'
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor='1.1', path=args.inPath)
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor='1.0')
		DH.splitTrainTest(fold_i=fold_i)
		
		testMiss = np.concatenate(DH.dataXmissingTest, axis=-1)
		idxMissTest = DH.missing_indices['test']
		testAcc= DH.dataXtest[0]
		test = np.concatenate(DH.dataXtest, axis=-1)
		trainX = np.concatenate(DH.dataXtrain, axis=-1)
		trainY = DH.dataYtrain
		yTrue = DH.dataYtest
		sm = SM()
		works, xRecMICE = sm.runMethod(testMiss,'MICE')
		# works, xRecMF = sm.runMethod(testMiss, 'matrixFactorization')
		# works, xRecEM = sm.runMethod(testMiss, 'expectationMaximization')
		# del sm
		# am = absoluteMetrics(testAcc,xRecMICE[:,:,0:3],idxMissTest)
		# res = am.runAll()
		# metricsMICE.append(res)
		model = load_model(os.path.join(classifiersPath,f'DCNN_acc_USCHAD_fold_{fold_i}.h5'))
		yPredMICE = model.predict(np.expand_dims(xRecMICE[:,:,0:3],axis = -1))
		yPredMICE = np.argmax(yPredMICE, axis=1)
		del am
		am = absoluteMetrics(testAcc,xRecMF[:,:,0:3],idxMissTest)
		res = am.runAll()
		metricsMF.append(res)
		yPredMF = model.predict(np.expand_dims(xRecMF[:,:,0:3],axis = -1))
		yPredMF = np.argmax(yPredMF, axis=1)
		del am
		am = absoluteMetrics(testAcc,xRecEM[:,:,0:3],idxMissTest)
		res = am.runAll()
		metricsEM.append(res)
		yPredEM = model.predict(np.expand_dims(xRecEM[:,:,0:3],axis = -1))
		yPredEM = np.argmax(yPredEM, axis=1)
		yPredTrue = model.predict(np.expand_dims(testAcc[:,:,0:3],axis = -1))
		yPredTrue = np.argmax(yPredTrue, axis=1)
		del am
		del DH
		for name,pred in zip(['MICE','MF','EM','Test'],[yPredMICE,yPredMF,yPredEM,yPredTrue]):
			classifResult[name + '_acc'].append(accuracy_score(yTrue, pred))
			classifResult[name + '_rec'].append(recall_score(yTrue, pred, average='macro'))
			classifResult[name + '_f1'].append(f1_score(yTrue, pred, average='macro'))

	metricsMICE = summarizeMetric(metricsMICE)
	metricsEM = summarizeMetric(metricsEM)
	metricsMF = summarizeMetric(metricsMF)
	resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))

	savePath = os.path.join(args.outPath, f'result_MICE_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(metricsMICE, write_file)
	savePath = os.path.join(args.outPath, f'result_EM_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(metricsEM, write_file)
	savePath = os.path.join(args.outPath, f'result_MF_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(metricsMF, write_file)
	savePath = os.path.join(args.outPath, f'result_Classification_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(resultClass, write_file)