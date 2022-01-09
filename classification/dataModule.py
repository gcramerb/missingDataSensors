from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader,  Dataset

import os, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
#from utils.data import categorical_to_int
from utils.dataHandler import dataHandler

class myDataset(Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
	
	def __len__(self):
		return len(self.Y)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return {'data': self.X[idx], 'label': self.Y[idx]}


class DM(LightningDataModule):
	def __init__(
			self,
			datasetName: str = "USCHAD.npz",
			path: str = None,
			n_classes: int = 12,
			batch_size: int = 64,
			sensorFactor: str = '1.1',
			num_workers: int = 1
	):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.n_classes = n_classes
		self.path = path
		self.datasetName = datasetName
		self.sensorFactor =sensorFactor


	def _setup(self, fold = 0,sensor = slice(0,6) ):
		
		DH = dataHandler()
		DH.load_data(dataset_name=self.datasetName,
		             sensor_factor=self.sensorFactor,
		             path=self.path)
		DH.splitTrainTest(fold_i=fold)
		#
		# y = categorical_to_int(y).astype('int')
		# self.Y = np.argmax(y, axis=1).astype('long')
        #                                    generator=torch.Generator().manual_seed(12270))

		Xtrain = np.concatenate(DH.dataXtrain,axis = -1)[:,None,:,sensor]
		Ytrain = DH.dataYtrain
		Xval = np.concatenate(DH.dataXtest,axis = -1)[:,None,:,sensor]
		Yval = DH.dataYtest
		self.dataTrain = myDataset(Xtrain, Ytrain)
		self.dataVal = myDataset(Xval,Yval)
		self.dataTest = myDataset(Xval, Yval)

	def train_dataloader(self):
		return DataLoader(
			self.dataTrain,
			shuffle=True,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			drop_last=True)
	
	def val_dataloader(self):
		return DataLoader(
			self.dataVal,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			drop_last=True)
	
	def test_dataloader(self):
		return DataLoader(self.dataTest,
		                  batch_size=self.batch_size,
		                  shuffle=True,
		                  num_workers=self.num_workers,
		                  drop_last=True)

def get_dataLoader(dataX,dataY,batch_size = 64):
	data = myDataset(dataX[:,None,:,:],dataY)
	return DataLoader(data,
	                  batch_size=batch_size,
	                  shuffle=True,
	                  num_workers=1,
	                  drop_last=True)
