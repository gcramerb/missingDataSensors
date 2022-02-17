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
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
parser.add_argument('--missingSensor', type=str, default="both")
args = parser.parse_args()

if args.slurm:
	args.inPath = '/storage/datasets/sensors/LOSO/'
	#args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
	args.outPath = os.path.abspath("/storage/datasets/HAR/Reconstructed/")

	if args.debug:
		import pydevd_pycharm
		pydevd_pycharm.settrace('172.22.100.3', port=22, stdoutToServer=True, stderrToServer=True, suspend=False)

else:
	args.inPath = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	args.outPath =  'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\Reconstructed\\'


args.outPath = os.path.join(args.outPath,args.missingSensor)
if __name__ == '__main__':
	metrics = []
	n_epoch = 90
	
	"""
	Sensor_idx: the sensor that gonna be reconstructed
	"""
	if args.missingSensor =='acc':
		missing_sensor = '1.0'
		sensor_idx = slice(0,1)
	elif args.missingSensor =='gyr':
		missing_sensor = '0.1'
		sensor_idx = slice(1,2)
	elif args.missingSensor =='both':
		missing_sensor = '1.1'
		sensor_idx = slice(0,2)
	start = time.time()
	
	recAEy_list = []
	y_list = []
	idx_list = []
	GT_list = []
	for fold_i in range(args.Nfolds):
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset,path = args.inPath, sensor_factor='1.1.0')
		DH.apply_missing(missingRate=args.missingRate, missing_sensor=missing_sensor)
		DH.impute('mean')
		DH.splitTrainTest(fold_i)
		
		train_data ,test_data = DH.get_data_pytorch(index=True,sensor_idx = sensor_idx)
		myModel = denoisingAEy(ms =missing_sensor)
		myModel.buildModel()
		hist = myModel.train(train_data,n_epoch,verbose=True)
		recAEy, GT, recMean, labels,idxMissTest = myModel.predict(test_data)
		
		GT_list.append(GT)
		recAEy_list.append(recAEy)
		y_list.append(labels)
		idx_list.append(idxMissTest)
	mR = str(int(float(args.missingRate) * 100))
	saveRec = os.path.join(args.outPath,f'USCHAD_recAEy{args.missingSensor}_miss{mR}.npz')
	np.savez(saveRec,rec =recAEy_list,y_list = y_list,idx = idx_list)
	end = time.time()
	print('time:  ',(end - start)/60)

