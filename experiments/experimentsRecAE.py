from sklearn.metrics import mean_squared_error
import numpy as np
from Autoencoder.convAE import denoisingAEy
import sys, os, argparse
sys.path.insert(0, "../")
from utils.dataHandler import dataHandler
from utils.metrics import absoluteMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--missingRate', type=str, default='0.2')
parser.add_argument('--Nfolds', type=int, default=14)
parser.add_argument('--dataset', type=str, default="USCHAD.npz")
args = parser.parse_args()
if args.slurm:
	args.inPath = '/storage/datasets/sensors/LOSO/'
	args.outPath = os.path.abspath("/home/guilherme.silva/missingDataSensors/results/")
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
	n_epoch = 1
	for fold_i in range(args.Nfolds):
		DH = dataHandler()
		DH.load_data(dataset_name=args.dataset, sensor_factor='1.1.0')
		DH.apply_missing(missing_factor=args.missingRate, missing_sensor='1.0')
		DH.impute('mean')
		DH.splitTrainTest(fold_i)
		# train_data ,test_data = DH.get_data_pytorch(index=True)
		train_data, test_data = DH.get_data_pytorch()
		myModel = denoisingAEy()
		myModel.buildModel()
		hist = myModel.train(train_data,n_epoch)
		recAEy, GT, recMean, labels = myModel.predict(test_data)
		del DH
		recMean = np.stack(recMean)
		am = absoluteMetrics(GT[:,:,0:3],recAEy[:,:,0:3])
		res = am.runAll()
		metricsAEy.append(res)
	metrics = absoluteMetrics.summarizeMetric(metricsAEy)
	savePath = os.path.join(args.outPath, f'result_AEy_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(metrics, write_file)

