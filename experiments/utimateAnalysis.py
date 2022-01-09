import numpy as np
import pandas as pd
import sys,os, argparse,json,time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import scipy.stats as st


sys.path.insert(0,"../utils/")
from dataHandler import dataHandler
from metrics import absoluteMetrics
sys.path.insert(0,"../classification/")
from dataModule import get_dataLoader
from trainer import clfDCNN

"""
This experiment aims to mensure the quality of The imputation of the three standatd methods.
We analyse both clssification and absolute metrics.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate', type=str, default='0.2')
parser.add_argument('--missingSensor', type=str, default='gyr')
parser.add_argument('--clfSensor', type=str, default='gyr')
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()

if args.slurm:
	sys.path.insert(0, "/home/guilherme.silva/missingDataSensors")
	inPathDataset = '/storage/datasets/sensors/LOSO/'
	args.inPath ="/storage/datasets/HAR/Reconstructed/"
	args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	classifiersPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/saved/")

	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)
else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\Reconstructed\\'
	inPathDataset = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\results\\"
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Github\\missingDataSensors\\saved\\")
	from plotGenerator import plot_result
	
	import seaborn as sns
	import pandas as pd
	import matplotlib.pyplot as plt
	
#from baselines.timeSeriesReconstruction import StandardMethods as SM
from utils.dataHandler import dataHandler
from utils.metrics import absoluteMetrics

def myMetric(data):
	m = np.mean(data)
	ic = st.t.interval(alpha=0.95, df=len(data) - 1, loc=m, scale=st.sem(data))
	return m,ic

def get_rec_data(fold_i):
	data = {}
	recFile = os.path.join(args.inPath, f'USCHAD_recAEy_{fold_i}_miss{args.missingRate}.npz')
	with np.load(recFile, allow_pickle=True) as tmp:
		data['acc'] = tmp['X']
		Yacc = tmp['y']
	mR = str(int(float(args.missingRate) * 100))
	recFile = os.path.join(args.inPath, 'gyr', f'USCHAD_recAEygyr_{fold_i}_miss{mR}_0.npz')
	with np.load(recFile, allow_pickle=True) as tmp:
		data['gyr'] = tmp['X']
		
	data['accGyr'] = np.concatenate([data['acc'],data['gyr']],axis = -1)
	return data[args.missingSensor]


classes = [ 'Walking Forward',
			'Walking Left',
			'Walking Right',
			'Walking Upstairs',
			'Walking Downstairs',
			'Running Forward',
			'Jumping Up',
			'Sitting',
			'Standing',
			'Sleeping',
			'Elevator Up',
			'Elevator Down']


if __name__ == '__main__':
	# process the data:
	metrics =[]
	classifResult = {}
	classifResult['AE_acc'] = []
	classifResult['AE_f1'] = []
	AE_cm = np.zeros([12,12])


	if args.missingSensor =='acc':
		ms = '1.0'
		s_idx = 0


	elif args.missingSensor =='gyr':
		ms = '0.1'
		s_idx = 1
	elif args.missingSensor =='accGyr':
		ms = '1.1'
		s_idx = slice(0,2)
		ini = 0
		end = 6
	if args.clfSensor =='all':
		clfSensor = 'accGyr'
		input_shape = args(1,500,6)
	else:
		clfSensor = args.missingSensor
		input_shape = (1,500,3)
	
	allRec = {}
	start = time.time()
	print('starting')
	for fold_i in range(args.Nfolds):

		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor='1.1', path=inPathDataset)
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor=ms)
		DH.splitTrainTest(fold_i=fold_i)
		
		testMiss = np.concatenate(DH.dataXmissingTest, axis=-1)
		#Reconstruction with standard methods
		#sm = SM()
		#allRec = sm.runAll(testMiss)
		
		allRec['AE'] = get_rec_data(fold_i)
		
		if args.missingSensor != clfSensor:
			oriSensor = DH.dataXtest[s_idx]
			if args.missingSensor == 'acc':
				allRec['AE'] = np.concatenate([allRec['AE'],oriSensor],axis = -1)
			else:
				allRec['AE'] = np.concatenate([oriSensor,allRec['AE']], axis=-1)
		yTrue = DH.dataYtest

		# ----------------- analysing the reconstruction - ----------------------"
		dataset = args.dataset.split('.')[0]
		file = f'{dataset}_fold_{fold_i}_{clfSensor}'
		model = clfDCNN(input_shape = input_shape)
		model.load_params(classifiersPath,file)
		
		for k,v in allRec.items():
			dl = get_dataLoader(v,yTrue)
			metrics = model.myTest(dl)
			classifResult[k + '_acc'].append(metrics['acc'])
			classifResult[k + '_f1'].append(metrics['f1'])
			AE_cm += metrics['cm']
		del testMiss, DH

	print('missing Rate:',args.missingRate,'\n')
	print('Missing Sensor:', args.missingSensor, '\n')
	print('Trial:', args.trial, '\n')
	
	resultClass = dict(map(lambda kv: (kv[0], myMetric(kv[1])), classifResult.items()))
	end = time.time()
	print(resultClass)
	print('\n')
	print(AE_cm)
	print('time:  ', (end - start) / 60)
	
	df_cm = pd.DataFrame(AE_cm, index=classes,
	                     columns=classes)
	fig, ax = plt.subplots(1, 1, figsize=(32, 12), dpi=80)
	sns.heatmap(df_cm, ax=ax, vmin=0, vmax=500, annot=True, fmt='.0f')
	ax.set_xlabel('True Label', fontsize=20)
	ax.set_ylabel('Pred Label', fontsize=20)
	ax.set_title(f'confusion matrix  - {args.missingSensor} rec', fontsize=30)
	plt.xticks(rotation=0)
	plt.show()
	plt.savefig(f'confusion matrix {args.missingSensor}_rec.png')
