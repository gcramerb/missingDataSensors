import pandas as pd
import numpy as np
import sys, os,random
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from math import log10, sqrt
from numpy.random import seed
from numpy.random import rand
sys.path.insert(0, "../utils/")
from activitiesNames import classesNames


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
		with np.load(data_input_file, allow_pickle=True) as tmp:
			X = tmp['X'].astype('float32')
			y_ = tmp['y']
			self.folds = tmp['folds']
		self.nClass = y_.shape[1]
		self.dataY = [np.argmax(i) for i in y_]
		self.dataYraw = y_
		

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
					temp.append(data[i])

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
					temp.append(data[i])

		if dataset_name == 'UTD-MHAD1_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(data[i])

		if dataset_name == 'UTD-MHAD2_1s.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(data[i])

		if dataset_name == 'WHARF.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(data[i])

		if dataset_name == 'USCHAD.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 0:3])  # ACC right-lower-arm
			data.append(X[:, :, :, 3:6])  # GYR right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(data[i])

		if dataset_name == 'WISDM.npz':
			data = []
			temp = []
			data.append(X[:, :, :, 3:6])  # ACC right-lower-arm
			s = sensor_factor.split('.')
			for i in range(len(s)):
				if s[i] == '1':
					temp.append(data[i])
		self.dataX = np.concatenate(temp, axis=1)

		
	def splitTrainTest(self,fold_i = None,ratio = 0.7,val = False):
		del self.dataXtest
		del self.dataXtrain
		self.dataXtrain = []
		self.dataXtest = []
		
		samples = self.dataX.shape[0]
		np.random.seed(0)
		if fold_i is None and val:
			trainSize = int(samples*(ratio - 0.1))
			valSize = int(samples*0.1)
			idx = np.random.permutation(samples)
			idx_train = idx[:trainSize]
			idx_val= idx[trainSize:valSize]
			idx_test =  idx[valSize:]
		elif fold_i is None:
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
			self.dataXtest = dataX[idx_test]
			self.dataXtrain  = dataX[idx_train]
			
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
			self.dataXmissingTest = dataXmissing[idx_test]
			self.dataXmissingTrain = dataXmissing[idx_train]
			
		if self.dataXreconstructed is not None:
			dataXreconstructed = deepcopy(self.dataXreconstructed)
			self.dataXreconstructedTest = dataXreconstructed[idx_test]
			self.dataXreconstructedTrain = dataXreconstructed[idx_train]
		
		if self.missing_indices is not None:
			aux = deepcopy(self.missing_indices)
			self.missing_indices = dict()
			self.missing_indices['train'] = aux[idx_train, :]
			self.missing_indices['test'] = aux[idx_test, :]


	def apply_missing(self,missingRate,missing_sensor = '1.0.0'):

		if self.dataX is None:
			raise ValueError("Dados inexistente")

		self.dataXmissing = deepcopy(self.dataX)
		nSamples = self.dataX.shape[0]
		dim = self.dataX.shape[2]
		s = missing_sensor.split('.')
		block_range = round(dim * float(missingRate))
		idx_range_max = dim - 2 - block_range
		idx_missing_all = []
		self.missing_indices = np.zeros((nSamples, block_range), dtype=np.int16)
		for i in range(0, nSamples):
			idx_missing = random.sample(range(0, idx_range_max), 1)[0]
			self.missing_indices[i, :] = range(idx_missing, idx_missing + block_range)
			for ms in range(len(s)):
				if s[ms] == '1':
						self.dataXmissing[i,ms, self.missing_indices[i, :] ,0:3] = np.nan

	def get_missing_indices(self):
		if self.missing_indices is not None:
			return self.missing_indices
		else:
			raise ValueError("ValueError: no missing indexes seted up")


	def impute(self,impute_type):
		self.dataXreconstructed = deepcopy(self.dataXmissing)
		nSamples,n_sensors,dim  = self.dataX.shape[0:3]
		self.imputeType = impute_type
		if  impute_type == 'mean':
			for i in range(nSamples):
				for j in range(n_sensors):
					idx_missing = np.argwhere(np.isnan(self.dataXreconstructed[i,j,:,0])) #All axis has the same missing points
					idx_missing = idx_missing.flatten()
					idx_notM = list(set(range(dim)) - set(idx_missing))
					defautMeanX = np.mean(self.dataXreconstructed[i,j, idx_notM, 0])
					defautMeanY = np.mean(self.dataXreconstructed[i,j, idx_notM, 1])
					defautMeanZ = np.mean(self.dataXreconstructed[i,j, idx_notM, 2])
					self.dataXreconstructed[i,j,idx_missing,0:3] = [defautMeanX,defautMeanY,defautMeanZ]

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
	

	def get_data_pytorch(self,index = False,sensor_idx =slice(0,2)):
		"""
		trainRec: both sensor contais missing acording to configuration
		train: the sensor the gonna be reconstructed
		testRec: both sensor contains missing acording to config
		"""
		train_data = []
		test_data = []
		all_index = self.get_missing_indices()
		train = self.dataXtrain[:,sensor_idx,:,:]
		if index:
			for i in range(len(train)):
				train_data.append([self.dataXreconstructedTrain[i], train[i],all_index['train'][i]] )
		else:
			for i in range(len(train)):
				train_data.append([trainRec[i], train[i], all_index['train'][i]])

		test = self.dataXtest[:,sensor_idx,:,:]
		for i in range(test.shape[0]):
			test_data.append([self.dataXreconstructedTest[i],test[i] , self.dataYtest[i],all_index['test'][i]])
		return train_data, test_data
