import pandas as pd
import numpy as np
import random
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_percentage_error
from math import log10, sqrt
from numpy.random import seed
from numpy.random import rand
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.realpath('../../../'))
from .activitiesNames import classesNames
from itertools import chain



class dataHandler():
	def __init__(self):
		self.dataX = None
		self.dataXtest =[]
		self.dataXtrain = []
		self.dataXmissing = None
		self.dataXmissingTest = []
		self.dataXreconstructed = None
		self.dataXreconstructedTest = []

		self.dataY = None
		self.dataYtest = None

		self.folds = None
		self.labelsNames = None
		self.nClass = None

		self.imputeType = None
		self.evalResult = None
		
		self.missing_indices = None

	def load_data(self,dataset_name, sensor_factor='1.0.0',path = None):
		if path is None:
			data_input_file = os.path.join('C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\', dataset_name)
		else:
			data_input_file = os.path.join(path,dataset_name)
		#data_input_file = '/home/guilherme.silva/datasets/LOSO/' + dataset_name
		tmp = np.load(data_input_file, allow_pickle=True)
		X = tmp['X'].astype('float32')
		y_ = tmp['y']
		self.nClass = y_.shape[1]
		self.dataY = [np.argmax(i) for i in y_]
		self.dataYraw = y_
		self.folds = tmp['folds']

		self.labelsNames = classesNames(dataset_name)

		if dataset_name == 'MHEALTH.npz':
			data = []
			temp = []
			data.append(X[:, 0, :, 14:17])  # ACC right-lower-arm
			# data.append(X[:, :, :, 5:8])  # ACC left-ankle sensor
			data.append(X[:, 0, :, 17:20])  # GYR right-lower-arm
			data.append(X[:, 0, :, 20:23])  # MAG right-lower-arm

			# data.append(X[:, :, :, 0:3])  # ACC chest-sensor
			# data.append(X[:, :, :, 5:8])  # ACC left-ankle sensor
			# data.append(X[:, :, :, 8:11])   # GYR left-ankle sensor
			# data.append(X[:, :, :, 11:14]) # MAG left-ankle sensor


			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'PAMAP2P.npz':
			data = []
			temp = []
			sensor_location = '3'
			if sensor_location == '1':
				data.append(X[:, 0, :, 1:4])  # ACC2 right-lower-arm
				data.append(X[:, 0, :, 7:10])  # GYR2 right-lower-arm
				data.append(X[:, 0, :, 10:13])  # MAG2 right-lower-arm
			if sensor_location == '2':
				data.append(X[:, :, :, 17:20])  # ACC2 right-lower-arm
				data.append(X[:, :, :, 20:23])  # GYR2 right-lower-arm
				data.append(X[:, :, :, 23:26])  # MAG2 right-lower-arm
			if sensor_location == '3':
				data.append(X[:, :, :, 27:30])  # ACC2 right-lower-arm
				data.append(X[:, :, :, 33:36])  # GYR2 right-lower-arm
				data.append(X[:, :, :, 36:39])  # MAG2 right-lower-arm
			s = sensor_factor.split('.')

			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'UTD-MHAD1_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'UTD-MHAD2_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'WHARF.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'USCHAD.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))

		if dataset_name == 'WISDM.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 3:6])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(np.squeeze(data[i]))
		Xsensor = np.concatenate(temp, axis=-1)
		self.dataX = temp
		
	def splitTrainTest(self,fold_i = None,ratio = 0.7,val = False):
		del self.dataXtest
		del self.dataXtrain
		self.dataXtrain = []
		self.dataXtest = []
		
		if fold_i is None and val:
			samples = len(self.dataX[0])
			np.random.seed(0)
			trainSize = int(samples*(ratio - 0.1))
			valSize = int(samples*0.1)
		
			idx = np.random.permutation(samples)
			idx_train = idx[:trainSize]
			idx_val= idx[trainSize:valSize]
			idx_test =  idx[valSize:]
		elif fold_i is None:
			samples = len(self.dataX[0])
			np.random.seed(0)
			max_ = int(samples*(ratio - 0.1))
			idx = np.random.permutation(samples)
			idx_train = idx[:max_]
			idx_test =  idx[max_:]
		else:
			idx_train = self.folds[fold_i][0]
			idx_test = self.folds[fold_i][1]
		
		
		
		if self.dataX is not None:
			dataX = deepcopy(self.dataX)
			self.dataXtrain = []
			for sensor in dataX:
				self.dataXtest.append(sensor[idx_test])
				self.dataXtrain.append(sensor[idx_train])
		if self.dataY is not None:
			dataY = deepcopy(self.dataY)
			self.dataYtrain = []
			self.dataYtrain = [dataY[i] for i in idx_train]
			self.dataYtest = [dataY[i] for i in idx_test]
		if self.dataYraw is not None:
			dataYraw = deepcopy(self.dataYraw)
			self.dataYrawTrain = []
			self.dataYrawTrain = [dataYraw[i] for i in idx_train]
			self.dataYrawTest = [dataYraw[i] for i in idx_test]
		if self.dataXmissing is not None:
			dataXmissing = deepcopy(self.dataXmissing)
			self.dataXmissingTrain = []
			for sensor in dataXmissing:
				self.dataXmissingTest.append(sensor[idx_test])
				self.dataXmissingTrain.append(sensor[idx_train])
		if self.dataXreconstructed is not None:
			dataXreconstructed = deepcopy(self.dataXreconstructed)
			self.dataXreconstructedTrain = []
			for sensor in dataXreconstructed:
				self.dataXreconstructedTest.append(sensor[idx_test])
				self.dataXreconstructedTrain.append(sensor[idx_train])
		if self.missing_indices is not None:
			aux = deepcopy(self.missing_indices)
			if type(aux) is not list:
				self.missing_indices = dict()
				
				self.missing_indices['train'] = aux[idx_train, :]
				self.missing_indices['test'] = aux[idx_test, :]


	def apply_missing(self,missing_factor,missing_type = 'b',missing_sensor = '1.0.0'):

		if self.dataX is None:
			print('Dados inexistente ')
			return
		self.dataXmissing = deepcopy(self.dataX)
		nSamples = self.dataX[0].shape[0]
		dim = self.dataX[0].shape[1]
		s = missing_sensor.split('.')

		for i in range(len(s)):
			if s[i] == '1':
				if missing_type == 'b':
					block_range = round(dim * float(missing_factor))
					idx_range_max = dim - 1 - block_range
					idx_missing_all = []
					self.missing_indices = np.zeros((nSamples, block_range),dtype = np.int16)
					for j in range(0,nSamples):
						idx_missing = random.sample(range(0, idx_range_max), 1)[0]
						self.dataXmissing[i][j, idx_missing:idx_missing + block_range, 0:3] = np.nan
						self.missing_indices[j,:] = range(idx_missing,idx_missing + block_range)

				if missing_type == 'nb':
					# usamos valor defaut de 3 partes ausentes
					# a princípo não está sendo tratado se os blocos faltantes forem sobrepostos.
					n = 3
					block_range = round(dim * float(missing_factor))
					block_i = round(block_range/n)
					dim_i = round(dim/n)
					idx_range_max = dim - 1 - block_i
					idx_range_max_i = round(idx_range_max/n)
					#self.missing_indices = np.zeros((nSamples, block_i*3))
					for j in range(nSamples):
						#aux_MIdx = []
						for k in range(0,n):
							ini_range = k*idx_range_max_i
							end_range = ini_range + idx_range_max_i
							idx_missing = random.sample(range(ini_range, end_range), 1)[0]
							self.dataXmissing[i][j, idx_missing:idx_missing + block_i, 0:3] = np.nan
							#aux_MIdx.append(range(idx_missing,idx_missing + block_i))
						#self.missing_indices[(i+1)*'1'][j,:] = aux_MIdx
				elif missing_type == 'u':
					idx_notMissing = list(range(0,dim,missing_factor))
					#idx_notMissing = idx_notMissing.flatten()
					self.missing_indices = list(set(range(dim)) - set(idx_notMissing))
					self.dataXmissing[i][:, self.missing_indices, 0:3] = np.nan

					
	def get_missing_indices(self):
		if self.missing_indices is not None:
			return self.missing_indices
		else:
			print('ERROR')
			return []

	def impute(self,impute_type):
		self.dataXreconstructed = deepcopy(self.dataXmissing)
		nSamples = self.dataX[0].shape[0]
		dim = self.dataX[0].shape[1]
		self.imputeType = impute_type
		if  impute_type == 'mean':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0])) #All axis has the same missing points
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim)) - set(idx_missing))
					defautMeanX = np.mean(sensor[i, idx_notM, 0])
					defautMeanY = np.mean(sensor[i, idx_notM, 1])
					defautMeanZ = np.mean(sensor[i, idx_notM, 2])
					sensor[i,idx_missing,0:3] = [defautMeanX,defautMeanY,defautMeanZ]
					#defautMeanX = np.mean(data_missing[i, idx_notM])
					#data_missing[i, idx_missing] = defautMeanX
		if  impute_type == 'RWmean':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed: # each sensor (acc,gyr..)
					for e in range(3): #axis of sensor
						ini_idx = 0
						first  = sensor[i,0,e]
						for t in range(1,dim):
							if not np.isnan(sensor[i,t,e]):
								end = sensor[i,t,e]
								mean = (first + end) / 2
								sensor[i, ini_idx:t, e] = mean
								first = end
								ini_idx = t
						sensor[i, ini_idx:, e] = mean
		if impute_type == 'default':
			defalt_values = [[0, 0, 0],[0, 0,0],[0, 0, 0]]
			for i in range(nSamples):
				j = 0
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					sensor[i, idx_missing, 0] = 0.0
					sensor[i, idx_missing, 1] = 0.0
					sensor[i, idx_missing, 2] = 0.0
					j = j+1
					#self.dataXmissing[i, idx_missing] = 0

		if impute_type == 'median':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim))- set(idx_missing))
					defautMedianX = np.median(sensor[i,idx_notM,0])
					defautMedianY = np.median(sensor[i,idx_notM,1])
					defautMedianZ = np.median(sensor[i,idx_notM,2])
					sensor[i,idx_missing,0:3] = [defautMedianX,defautMedianY,defautMedianZ]

		if impute_type == 'last_value':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					if idx_missing.shape[0] > 0:
						idx_notM = list(set(range(dim)) - set(idx_missing))
						lastVx = sensor[i,idx_missing[0]-1,0]
						lastVy = sensor[i,idx_missing[0]-1,1]
						lastVz = sensor[i,idx_missing[0]-1,2]
						sensor[i, idx_missing, 0:3] = [lastVx, lastVy, lastVz]
		if impute_type == 'aleatory':
			# nao sera usado
			seed(22277)
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					n = len(idx_missing)
					minX = np.nanmin(sensor[i,:,0])
					minY = np.nanmin(sensor[i,:,1])
					minZ = np.nanmin(sensor[i,:,2])

					maxX = np.nanmax(sensor[i,:,0])
					maxY= np.nanmax(sensor[i,:,1])
					maxZ =np.nanmax(sensor[i,:,2])
					x = minX + (rand(n) * (maxX - minX))
					y = minY + (rand(n) * (maxY - minY))
					z = minZ + (rand(n) * (maxZ - minZ))
					sensor[i, idx_missing, 0:3] = [x,y,z]

		if impute_type == 'interpolation':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					sensor[i,:,0 ] = pd.Series(sensor[i,:,0 ]).interpolate()
					sensor[i,:, 1] = pd.Series(sensor[i, :, 1]).interpolate()
					sensor[i,:,2] = pd.Series(sensor[i, :, 2]).interpolate()



		if impute_type == 'frequency':
			for i in range(nSamples):
				for sensor in self.dataXreconstructed:
					idx_missing = np.argwhere(np.isnan(sensor[i,:,0]))
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim)) - set(idx_missing))
					xfreq =  fftpack.rfft(sensor[i, idx_notM, 0])
					yfreq =fftpack.rfft(sensor[i, idx_notM, 1])
					zfreq =fftpack.rfft(sensor[i, idx_notM, 2])

					sensor[i,idx_missing,0] = fftpack.irfft(xfreq, n=len(idx_missing))
					sensor[i, idx_missing, 1] = fftpack.irfft(yfreq, n=len(idx_missing))
					sensor[i, idx_missing, 2] = fftpack.irfft(zfreq, n=len(idx_missing))
	

	def get_data_keras(self,index = False):

		X_trainRec = self.dataXreconstructedTrain
		X_train = self.dataXtrain
		X_testRec = self.dataXreconstructedTest
		
		# X_GT = DH.dataXtest
		trainRec = []
		train = []
		testRec = []
		#
		for sTrainRec, sTrain in zip(X_trainRec, X_train):
			s0 = np.expand_dims(sTrainRec, axis=1)
			s1 = np.expand_dims(sTrain, axis=1)
			trainRec.append(s0)
			train.append(s1)
		for sTest in X_testRec:
			s2 = np.expand_dims(sTest, axis=1)
			testRec.append(s2)
		if index:
			return train,trainRec,testRec, self.get_missing_indices()
		else:
			return train, trainRec, testRec
	
	def get_data_pytorch(self,index = False):
		train, trainRec, testRec, all_index = self.get_data_keras(index)

		if index:
			train_data = []
			for i in range(len(train[0])):
				train_data.append([[xr[i] for xr in trainRec], train[0][i],all_index['train'][i]] )
			
			test_data = []
			for i in range(len(testRec[0])):
				test_data.append(
					[[x[i] for x in testRec], np.expand_dims(self.dataXtest[0][i], axis=0), self.dataYtest[i],all_index['test'][i]])
			return train_data, test_data
		else:
			train_data = []
			for i in range(len(train[0])):
				train_data.append([[xr[i] for xr in trainRec], train[0][i]])
			test_data = []
			for i in range(len(testRec[0])):
				test_data.append(
					[[x[i] for x in testRec], np.expand_dims(self.dataXtest[0][i], axis=0), self.dataYtest[i]])
			return train_data, test_data
	
	def get_data_reconstructed(self,dataset_name,miss,imp,si,path,file,fold):
		if si:
			self.load_data(dataset_name=dataset_name, sensor_factor='1.0', path=path)
			self.apply_missing(missing_factor=miss, missing_sensor='1.0')
			self.impute(imp)
			self.splitTrainTest(fold_i=fold_i)
			
			testRec = self.dataXreconstructedTest[0]
			testTrue = self.dataXtest[0]
			# y_true = DH.dataYtest
			idxAll = self.get_missing_indices()['test']
		else:
			data = np.load(path +file, allow_pickle=True)
			testRec = data['deploy_data']
			self.load_data(dataset_name=dataset_name, sensor_factor='1.0', path=path)
			self.splitTrainTest(fold_i=fold)
			testTrue = self.dataXtest[0]
			# test = data['data']
			y_true = data['classes']
			idxAll = None
		return testRec, testTrue, idxAll
	
"""path = 'C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'
DH = dataHandler()
DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.1', path=path)
DH.apply_missing(missing_factor='0.2', missing_sensor='1.0')
DH.impute('mean')
DH.splitTrainTest(fold_i=fold_i)"""
