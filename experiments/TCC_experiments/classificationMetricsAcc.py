import numpy as np
from keras.models import load_model
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import scipy.stats as st
import sys
import pandas as pd
import json
import os
import matplotlib

matplotlib.use('tkagg')

import matplotlib.pyplot as plt

sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler
from utils import saveAll

if __name__ == '__main__':
	np.random.seed(12227)
	if len(sys.argv) > 1:
		data_input_file = sys.argv[1]
	else:
		data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
		path = data_input_file
	dataset_name = 'USCHAD.npz'
	dataset = dataset_name.split('.')[0]
	
	default_impute = ['mean']
	# default_impute = []
	impute_list = []
	impute_list = []
	impute_list = impute_list + default_impute
	Result = dict()
	n_folds = 14
	models_list = []
	for fold_i in range(n_folds):
		model_name = 'USCHAD_fold_' + str(fold_i) + '.h5'
		model = load_model('models/' + model_name)

		result_file = dataset + '_' + '00' + '_NoImpute'
		avg_acc = []
		avg_recall = []
		avg_f1 = []
		avg_cm = []
		
		DH = dataHandler()
		DH.load_data(dataset_name=dataset_name, sensor_factor='1.0', path=path)
		DH.splitTrainTest(fold_i=fold_i)
		
		test = DH.dataXtest[0]
		y_true = DH.dataYtest
		
		test = np.expand_dims(test, axis=1)
		y_pred = model.predict(test)
		y_pred = np.argmax(y_pred, axis=1)
		
		acc_fold = accuracy_score(y_true, y_pred)
		avg_acc.append(acc_fold)
		recall_fold = recall_score(y_true, y_pred, average='macro')
		avg_recall.append(recall_fold)
		f1_fold = f1_score(y_true, y_pred, average='macro')
		avg_f1.append(f1_fold)
		
		cm_fold = confusion_matrix(y_true, y_pred)
		avg_cm.append(cm_fold)
	ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
	ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
	ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
	
	result_name = f'Sena_ACC_' + result_file + '.json'
	Result = dict()
	Result['acuracia'] = np.mean(avg_acc)
	Result['reacall'] = np.mean(avg_recall)
	Result['F1'] = np.mean(avg_f1)
	Result['IC acuracia'] = ic_acc
	Result['IC recall'] = ic_recall
	Result['IC F1'] = ic_f1
	confusionMatrix = np.sum(avg_cm, axis=0) / n_folds
	path_result = os.path.realpath('metrics')
	result_file_name = os.path.join(path_result, result_name)
	with open(result_file_name, "w") as write_file:
		json.dump(Result, write_file)
	cm_name = result_file_name.replace('json', 'npz')
	cm_name = cm_name.split('/')[-1]
	np.savez(cm_name, cm=confusionMatrix)


