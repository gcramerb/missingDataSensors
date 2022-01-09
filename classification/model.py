import numpy as np
import pandas as pd
import pickle, sys
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st


import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNNclassifier(nn.Module):
	def __init__(self,input_shape = (1,500,6)):
		super(DCNNclassifier, self).__init__()
		self.n_classes= 12
		self.width = (16, 32)
		self.kernel_pool = [(2, 2), (3, 3), (5, 2)]
		self.n_neurons = 50
		self.input_shape = input_shape
		self.linerNeu = self.input_shape[-1] *4000
		#self.n_neurons = 20
		self.dropout_rate = 0.25
		self.kernel_init_dense = 'glorot_normal'

	
	def init_weights(self,m):
		if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight)
			m.bias.data.fill_(0.01)
	
	def _stream(self, kernel):
		in_channel = self.input_shape[0]
		seq = nn.Sequential(
			nn.Conv2d(in_channels=in_channel,
			          kernel_size=kernel,
			          out_channels=self.width[0],
			          padding='same', bias=True),
			nn.ReLU(),
			nn.MaxPool2d((2, 1)),
			nn.Conv2d(in_channels=self.width[0],
			          kernel_size=kernel,
			          out_channels=self.width[1],
			          padding='same', bias=True),
			nn.ReLU(),
			nn.MaxPool2d((2, 1)),
			nn.Flatten(),
			nn.Linear(self.linerNeu, self.n_neurons),
			nn.SELU(),
			nn.BatchNorm1d(self.n_neurons),
			nn.Dropout(self.dropout_rate),
			nn.Linear(self.n_neurons, self.n_classes),
			nn.Softmax(dim=1)
		)
		return seq
		
	def build(self):
		self.streams_models = nn.ModuleList([])
		for i in range(len(self.kernel_pool)):
			self.streams_models.append(self._stream(self.kernel_pool[i]))
		
		self.finalLayer = nn.Sequential(
			nn.Linear(len(self.kernel_pool) * self.n_classes, self.n_classes),
			nn.Softmax(dim=1)
		)
		
	def forward(self,X):
		out = []
		for streams_i in self.streams_models:
			out.append(streams_i(X))

		if len(out) > 1:
			out = torch.cat(out, -1)
			out = self.finalLayer(out)
		else:
			out = out[0]
		return out
		
