import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st
import os, sys,json
from dataHandler import dataHandler
import pickle

sys.path.insert(0, "C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\")
from Catal import Catal


def trainSaveClassifiers(datasetName, inPath, outPath):
	# dataPreparation:
	tmp = np.load(os.path.join(inPath, f'{datasetName}.npz'), allow_pickle=True)
	X = tmp['X']
	X = X[:, 0, :, :]
	y = tmp['y']
	folds = tmp['folds']
	n_class = y.shape[1]
	y = np.argmax(y, axis=1)
	# - ------------------------
	
	for i in range(0, len(folds)):
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		X_train = X[train_idx]
		# concatenate gyr + acc_reconstructed:
		catal_classifier = Catal()
		catal_classifier.fit(X_train, y[train_idx])
		with open(os.path.join(outPath, f'Catal_{datasetName}_{i}.pkl'), 'wb') as output:
			pickle.dump(catal_classifier, output, pickle.HIGHEST_PROTOCOL)
		del catal_classifier


def classificationResult(data, Fold, path):
	with open(os.path.join(path, f'Catal_USCHAD_{Fold}.pkl'), 'rb') as input:
		catal_classifier = pickle.load(input)
	yPred = catalClassifier.predict(data)
	return yPred


def catalAllClassification():
	dataset = 'USCHAD.npz'
	missing_list = ['0.5']
	# missing_list = ['0.2']
	finalResult = dict()
	finalResult['acc'] = dict()
	finalResult['f1'] = dict()
	finalResult['rec'] = dict()
	for miss in missing_list:
		finalResult['acc'][miss] = list()
		finalResult['f1'][miss] = list()
		finalResult['rec'][miss] = list()
	
	# data_input_file = path + dataset
	data_input_file = os.path.abspath('C:\\Users\\gcram\\Documents\\Datasets\\USCHAD_forBRITS\\')
	classifier = "Catal"
	
	simple_impute = False
	# imputeList = ['mean']
	# imputeList = ['AEY_mse']
	
	# dataPreparation:
	X = tmp['X']
	X = X[:, 0, :, :]
	y = tmp['y']
	folds = tmp['folds']
	n_class = y.shape[1]
	
	avg_acc = []
	avg_recall = []
	avg_f1 = []
	y = np.argmax(y, axis=1)
	# - ------------------------
	# Result :
	acc = dict()
	f1 = dict()
	recall = dict()
	
	for i in range(0, len(folds)):
		xRec_list = []
		yRec_list = []
		
		train_idx = folds[i][0]
		test_idx = folds[i][1]
		X_train = X[train_idx]
		
		# concatenate gyr + acc_reconstructed:
		gyr = X[test_idx][:, :, 3:6]
		for miss in missing_list:
			if simple_impute:
				DH = dataHandler()
				DH.load_data(dataset_name=dataset, sensor_factor='1.0', path=path)
				DH.apply_missing(missing_factor=miss, missing_sensor='1.0')
				DH.impute('mean')
				DH.splitTrainTest(fold_i=i)
				testRec = DH.dataXreconstructedTest[0]
				xRec = np.concatenate([testRec, gyr], axis=-1)
				yRec = DH.dataYtest
				yRec_list.append(yRec)
				xRec_list.append(xRec)
				del DH
			else:
				file = 'USCHAD' + '_' + miss + '_AEY_mse' + str(i) + '.npz'
				data = np.load(path + file, allow_pickle=True)
				testRec = data['deploy_data']
				# test = data['data']
				yRec_list.append(data['classes'])
				xRec = np.concatenate([testRec, gyr], axis=-1)
				xRec_list.append(xRec)
		
		catal_classifier = Catal()
		catal_classifier.fit(X_train, y[train_idx])
		
		for i in range(len(missing_list)):
			miss = missing_list[i]
			y_pred = catal_classifier.predict(xRec_list[i])
			finalResult['acc'][miss].append(accuracy_score(yRec_list[i], y_pred))
			finalResult['f1'][miss].append(f1_score(yRec_list[i], y_pred, average='macro'))
			finalResult['rec'][miss].append(recall_score(yRec_list[i], y_pred, average='macro'))
	
	Result = dict()
	Result['acc'] = dict()
	Result['recall'] = dict()
	Result['f1'] = dict()
	
	for miss in missing_list:
		ic_acc = st.t.interval(0.9, len(finalResult['acc'][miss]) - 1, loc=np.mean(finalResult['acc'][miss]),
		                       scale=st.sem(finalResult['acc'][miss]))
		ic_recall = st.t.interval(0.9, len(finalResult['rec'][miss]) - 1, loc=np.mean(finalResult['rec'][miss]),
		                          scale=st.sem(finalResult['rec'][miss]))
		ic_f1 = st.t.interval(0.9, len(finalResult['f1'][miss]) - 1, loc=np.mean(finalResult['f1'][miss]),
		                      scale=st.sem(finalResult['f1'][miss]))
		
		Result['acc'][miss] = 'Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(finalResult['acc'][miss]),
		                                                                         ic_acc[0], ic_acc[1])
		Result['recall'][miss] = 'Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(finalResult['rec'][miss]),
		                                                                          ic_recall[0], ic_recall[1])
		Result['f1'][miss] = 'Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(finalResult['f1'][miss]), ic_f1[0],
		                                                                  ic_f1[1])
	
	path_result = os.path.realpath('../results/Catal')
	result_file_name_acc = os.path.join(path_result, f'{classifier}_resultsAccGyr_final.json')
	with open(result_file_name_acc, "w") as write_file:
		json.dump(Result, write_file)
	

if __name__ == '__main__':
	#Paper: On the use of ensemble of classifiers for accelerometer-based activity recognition
	np.random.seed(12227)
	
	if (len(sys.argv) > 1):
	    data_input_file = sys.argv[1]
	else:
	    path =  'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
	
	inPath = os.path.abspath('C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\')
	outPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Smart Sense\\HAR_classifiers\\trained\\')
	dataset = 'USCHAD'
	trainSaveClassifiers(dataset,inPath,outPath)
	
	
        