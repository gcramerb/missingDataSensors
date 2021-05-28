from sklearn.metrics import mean_squared_error
import numpy as np
from dataHandler import dataHandler
import os
import json

def myPreprocess():
	#out file
	init_path =os.path.abspath('C:\\Users\\gcram\\Documents\\Datasets\\USCHAD_forBRITS\\')
	output = os.path.join(init_path,'json')
	fs = open(output, 'w')
	
	## getting the data:
	m = '0.5'
	DH = dataHandler()
	DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.1.0')
	missing_sensor = '1.0'
	missing_axis = np.sum(3 *  list(map(int, missing_sensor.split('.')[:])))
	DH.apply_missing(missing_factor=m, missing_sensor=missing_sensor)
	DH.impute('default')
	DH.splitTrainTest(ratio = 0.7)
	
	jsonOutFile = {}
	
	#parse train
	#Merge sensors:
	xTrainRec = np.concatenate([DH.dataXreconstructedTrain[0],DH.dataXreconstructedTrain[1]],axis = -1)
	xTrain = np.concatenate([DH.dataXtrain[0],DH.dataXtrain[1]],axis = -1)
	y_train = DH.dataY
	idx = DH.get_missing_indices()
	shapes = xTrain.shape

	
	x_trainComp = np.concatenate([DH.dataXtrain[0],DH.dataXtrain[0]],axis = -1)
	for i in range(shapes[0]):
		sampleDict = dict()

		sampleDict['label'] = int(y_train[i])
		
		aux = dict()
		aux['values'] = xTrainRec[i].tolist()
		aux['masks'] = np.ones([shapes[1],shapes[2]])
		aux['masks'][idx['train'][i]] =np.concatenate([np.zeros(missing_axis),np.ones(shapes[-1]-missing_axis)])
		
		aux['eval'] = xTrain[i].tolist()
		aux['eval_masks'] = np.ones([shapes[1],shapes[2]]).tolist()
		aux['forward'] = None
		aux['deltas'] = np.zeros([shapes[1],shapes[2]])
		for j in range(1,shapes[1]):
			value = j +  np.multiply(aux['deltas'][j-1], ((aux['masks'][j-1]*-2)+2)/2)
			aux['deltas'][j] = value
		aux['deltas'] = aux['deltas'].tolist()
		aux['masks'] = aux['masks'].tolist()
		sampleDict['forward'] = aux
		

		#backward:
		aux =dict()

		aux['values'] = xTrainRec[i][::-1].tolist()
		aux['masks'] = np.ones([shapes[1], shapes[2]])
		aux['masks'][shapes[1] - idx['train'][i]-1] = np.concatenate([np.zeros(missing_axis), np.ones(shapes[-1] - missing_axis)])
		aux['eval'] = xTrain[i][::-1].tolist()
		aux['eval_masks'] = np.ones([shapes[1], shapes[2]]).tolist()
		aux['forward'] = None
		aux['deltas'] = np.zeros([shapes[1], shapes[2]])
		for j in range(1, shapes[1]):
			value = j + np.multiply(aux['deltas'][j - 1], ((aux['masks'][j - 1] * -2) + 2) / 2)
			aux['deltas'][j] = value
		aux['deltas'] = aux['deltas'].tolist()
		aux['masks'] = aux['masks'].tolist()
		sampleDict['backward'] = aux
		jsonOutFile = json.dumps(sampleDict)
	
		fs.write(jsonOutFile + '\n')


	
	