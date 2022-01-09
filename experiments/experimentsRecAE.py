from sklearn.metrics import mean_squared_error
import numpy as np
import sys, os, argparse, json, time
sys.path.insert(0, "../Autoencoder/")
from convAE import denoisingAEy
sys.path.insert(0, "../utils/")
from dataHandler import dataHandler
from metrics import absoluteMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate', type=str, default='0.2')
parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
parser.add_argument('--sensor', type=str, default="both")
args = parser.parse_args()
if args.slurm:
	args.inPath = '/storage/datasets/sensors/LOSO/'
	#args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	args.outPath = os.path.abspath("/storage/datasets/HAR/Reconstructed/")
	classifiersPath = os.path.abspath("/home/guilherme.silva/classifiers/trained/")

	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath = "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\results\\"
	classifiersPath = os.path.abspath("C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\")

if __name__ == '__main__':
	metricsAEy = []
	n_epoch = 70
	if args.sensor =='acc':
		missing_sensor = '1.0'
		sensor_idx = 0
	elif args.sensor =='gyr':
		missing_sensor = '0.1'
		sensor_idx = 1
	elif args.sensor =='both':
		missing_sensor = '1.1'
		sensor_idx = 1
	start = time.time()
	
	recAEy_list = []
	y_list = []
	idx_list = []
	GT_list = []
	for fold_i in range(args.Nfolds):
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset,path = args.inPath, sensor_factor='1.1.0')
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor=missing_sensor)
		DH.impute('mean')
		DH.splitTrainTest(fold_i)
		
		train_data ,test_data = DH.get_data_pytorch(index=True,sensor_idx = 0)
		myModel = denoisingAEy()
		myModel.buildModel()
		hist = myModel.train(train_data,n_epoch,verbose=True)
		recAEy, GT, recMean, labels,idxMissTest = myModel.predict(test_data)
		
		GT_list.append(GT)
		recAEy_list.append(recAEy)
		y_list.append(labels)
		idx_list.append(idxMissTest)
	mR = str(int(float(args.missingRate) * 100))

	saveRec = os.path.join(args.outPath,args.sensor,f'USCHAD_recAEy{args.sensor}_miss{mR}_{args.trial}.npz')
	np.savez(saveRec,rec =recAEy_list,y_list = y_list,GT = GT_list,idx = idx_list)
	end = time.time()
	print('time:  ',(end - start)/60)

