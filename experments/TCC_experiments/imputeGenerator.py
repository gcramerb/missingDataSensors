from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import os
import json


if (len(sys.argv) > 1):
	data_input_file = sys.argv[1]
	path = '/storage/datasets/HAR/LOSO/'
else:
	data_input_file = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
	path = data_input_file
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\Autoencoder_creation\\")
	sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")

#test = path + 'USCHAD_0.2_AE_mse.npz'
#a = np.load(test,allow_pickle = True)
from dataHandler import dataHandler


missing_list = ['0.3','0.4','0.5','0.6','0.7']
dataset_name = 'USCHAD.npz'
sensor_factor = '1.1'

DH = dataHandler()
DH.load_data(dataset_name=dataset_name, sensor_factor=sensor_factor, path=path)
DH.apply_missing(missing_factor=2,missing_type = 'u', missing_sensor='1.0')
DH.impute('RWmean')
DH.splitTrainTest(fold_i=fold_i)
deploy_data_all = DH.dataXreconstructedTest[0]
y_all = DH.dataYtest





impute_list = ['mean','defaut','last_value','interpolation']
for miss in missing_list:
	for imp in impute_list:
		dataRec = dict()
		classes = dict()
		for fold_i in range(14):
			DH = dataHandler()
			DH.load_data(dataset_name=dataset_name, sensor_factor=sensor_factor, path=path)
			DH.apply_missing(missing_factor=miss, missing_sensor='1.0')
			DH.impute(imp)
			DH.splitTrainTest(fold_i=fold_i)
			deploy_data_all = DH.dataXreconstructedTest[0]
			y_all = DH.dataYtest
			del DH
			dataset = dataset_name.split('.')[0]
			path_out_name = path + dataset+'_' + miss + '_' + imp +str(fold_i) +'.npz'
			## as amostras mudam de ordem ?
			# pq o y sai com valor diferente do test ?
			np.savez(path_out_name,deploy_data = deploy_data_all,classes = y_all)
