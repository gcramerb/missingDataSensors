from sklearn.metrics import mean_squared_error
import numpy as np
from dataHandler import dataHandler
import os
import json


class dataGenerator:
	def __init__(self,dataset = 'USCHAD.npz',missing_sensor = '1.0',missing = '0.5'):
		self.dataset = dataset
		self.missing = missing
		self.missing_sensor = missing_sensor
		self.inPath = None
		self.outPath = None
	def setPath(self, inPath = None,outPath = None):
		if inPath is None:
			self.outPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Datasets\\USCHAD_forBRITS\\')
			self.inPath = None
		else:
			self.inPath = inPath
			self.outPath = outPath
	def process(self,x,xRec,idx,missing_axis):
		shapes  = x.shape
		auxForward = dict()
		auxForward['values'] = xRec.tolist()
		auxForward['masks'] = np.ones([shapes[0], shapes[1]])
		auxForward['masks'][idx] = np.concatenate([np.zeros(missing_axis), np.ones(shapes[-1] - missing_axis)])
		auxForward['evals'] = x.tolist()
		auxForward['eval_masks'] = np.ones([shapes[0], shapes[1]]).tolist()
		auxForward['forwards'] =  xRec.tolist()
		auxForward['deltas'] = np.zeros([shapes[0], shapes[1]])
		for j in range(1, shapes[0]):
			value = j + np.multiply(auxForward['deltas'][j - 1], ((auxForward['masks'][j - 1] * -2) + 2) / 2)
			auxForward['deltas'][j] = value
		auxForward['deltas'] = auxForward['deltas'].astype('int32').tolist()
		auxForward['masks'] = auxForward['masks'].astype('int32').tolist()
		
		auxBackward = dict()
		auxBackward['values'] = xRec[::-1].tolist()
		auxBackward['masks'] = np.ones([shapes[0], shapes[1]])
		auxBackward['masks'][shapes[1] - idx -1] = np.concatenate([np.zeros(missing_axis), np.ones(shapes[-1] - missing_axis)])
		auxBackward['evals'] = x[::-1].tolist()
		auxBackward['eval_masks'] = np.ones([shapes[0], shapes[1]]).tolist()
		auxBackward['forwards'] =  xRec.tolist()
		auxBackward['deltas'] = np.zeros([shapes[0], shapes[1]])
		for j in range(1, shapes[1]):
			value = j + np.multiply(auxBackward['deltas'][j - 1], ((auxBackward['masks'][j - 1] * -2) + 2) / 2)
			auxBackward['deltas'][j] = value
		auxBackward['deltas'] = auxBackward['deltas'].astype('int32').tolist()
		auxBackward['masks'] = auxBackward['masks'].astype('int32').tolist()
		return auxForward ,auxBackward
			
	def myPreprocess(self,fold = 0,save = True):
		
		processedFile = self.dataset.split('.')[0] +'_'+self.missing+ f'_fold_{fold}'
		outputTrain = os.path.join(self.outPath,processedFile + '_train')
		outputTest = os.path.join(self.outPath, processedFile + '_test')
		DH = dataHandler()
		DH.load_data(dataset_name=self.dataset, sensor_factor='1.1.0',path =self.inPath )
		## getting the data:
		missing_axis = np.sum(3 *  list(map(int, self.missing_sensor.split('.')[:])))
		DH.apply_missing(missing_factor=self.missing, missing_sensor=self.missing_sensor)
		DH.impute('default')
		DH.splitTrainTest(fold_i = fold)
		
		jsonOutFileTrain = {}
		jsonOutFileTest ={}
		
		#parse train
		#Merge sensors:
		xTrainRec = np.concatenate([DH.dataXreconstructedTrain[0],DH.dataXreconstructedTrain[1]],axis = -1)
		xTrain = np.concatenate([DH.dataXtrain[0],DH.dataXtrain[1]],axis = -1)
		y_train = DH.dataY
		idx = DH.get_missing_indices()
		shapesTrain = xTrain.shape
		
		xTestRec = np.concatenate([DH.dataXreconstructedTest[0], DH.dataXreconstructedTest[1]], axis=-1)
		xTest = np.concatenate([DH.dataXtest[0], DH.dataXtest[1]], axis=-1)
		y_test = DH.dataYtest
		shapesTest = xTest.shape
		
		if save:
			fsTrain = open(outputTrain, 'w')
			fsTest= open(outputTest, 'w')

			for i in range(shapesTrain[0]):
				sampleDict = dict()
				sampleDict['label'] = int(y_train[i])
				forward , backward = self.process(xTrain[i],xTrainRec[i],idx['train'][i],missing_axis)
				sampleDict['forward'] = forward
				sampleDict['backward'] = backward
				jsonOutFileTrain = json.dumps(sampleDict)
				fsTrain.write(jsonOutFileTrain + '\n')
			for i in range(shapesTest[0]):
				sampleDict = dict()
				sampleDict['label'] = int(y_test[i])
				forward, backward = self.process(xTest[i], xTestRec[i], idx['test'][i], missing_axis)
				sampleDict['forward'] = forward
				sampleDict['backward'] = backward
				jsonOutFileTest = json.dumps(sampleDict)
				fsTest.write(jsonOutFileTest + '\n')
			fsTest.close()
			fsTrain.close()
				
		else:
			jsonOutFileTest = []
			jsonOutFileTrain = []
			for i in range(shapesTrain[0]):
				sampleDict = dict()
				sampleDict['label'] = int(y_train[i])
				forward , backward = self.process(xTrain[i],xTrainRec[i],idx['train'][i],missing_axis)
				sampleDict['forward'] = forward
				sampleDict['backward'] = backward
				jsonOutFileTrain.append(json.dumps(sampleDict))
			for i in range(shapesTest[0]):
				sampleDict = dict()
				sampleDict['label'] = int(y_test[i])
				forward, backward = self.process(xTest[i], xTestRec[i], idx['test'][i], missing_axis)
				sampleDict['forward'] = forward
				sampleDict['backward'] = backward
				jsonOutFileTest.append(json.dumps(sampleDict))
			return jsonOutFileTrain,jsonOutFileTest
		return None,None

		
