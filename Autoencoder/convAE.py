import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys, pickle
import numpy as np
from copy import deepcopy

sys.path.insert(0, "../Autoencoder/")
from modelUtils.custom_losses import SoftDTW
from modelUtils.custom_losses import My_dct_Loss as DCT_loss

# define the NN architecture
class ConvAutoencoder(nn.Module):
	def __init__(self,hyp = None):
		super(ConvAutoencoder, self).__init__()
		if hyp:
			conv_window = hyp['conv_window']
			pooling_window_1 = hyp['pooling_window_1']
			same_pad = 	pooling_window_1
			pooling_window_2 = hyp['pooling_window_2']
			n_filters = hyp['n_filters']
			encoded_dim = hyp['encDim']
	
		else:
			conv_window = (5,3)
			same_pad = (2,1)
			pooling_window_1 = (2,1)
			pooling_window_2 = (5, 1)
			n_filters = (32, 16, 8)
			encoded_dim = n_filters[1]

		## encoder layers ##
		self.conv1 = nn.Conv2d(in_channels = 1,kernel_size = conv_window,out_channels = n_filters[0], padding=same_pad)
		self.pool1 = nn.MaxPool2d(pooling_window_1)
		self.conv2 = nn.Conv2d(in_channels = n_filters[0],kernel_size=conv_window, out_channels=n_filters[1], padding=same_pad)
		self.pool2 = nn.MaxPool2d(pooling_window_2)
		self.encoded = nn.Conv2d(in_channels = n_filters[1],kernel_size=conv_window, out_channels=n_filters[2], padding=same_pad)

		## decoder layers ##
		
		self.conv_neg_3 = nn.Conv2d(in_channels = encoded_dim ,kernel_size=conv_window, out_channels=n_filters[1], padding=same_pad)
		self.up3 = nn.Upsample(scale_factor=pooling_window_2)
		self.conv_neg_2 = nn.Conv2d(in_channels = n_filters[1],kernel_size=conv_window, out_channels=n_filters[0], padding=same_pad) # mudar para sair 16 ?
		self.up2 = nn.Upsample(scale_factor=pooling_window_1)
		self.decoded = nn.Conv2d(in_channels =n_filters[0],kernel_size=conv_window, out_channels=1, padding=same_pad)

	def forward(self,X):
		encoded = []
		for sensor_data in X:
			encoded.append(self.encode(sensor_data))

		encoded = torch.cat(tuple(encoded),1)
		out = self.decode(encoded)
		return out

	def encode(self,inp):
		## encode ##

		x1 = F.relu(self.conv1(inp))
		x2 = self.pool1(x1)
		x3 = F.relu(self.conv2(x2))
		x4 = self.pool2(x3)
		encoded = F.relu(self.encoded(x4))
		return encoded
	
	def decode(self,encoded):
		## decode ##
		d1 = F.relu(self.conv_neg_3(encoded))
		d2 = self.up3(d1)
		d3 = F.relu(self.conv_neg_2(d2))
		d4 = self.up2(d3)
		decoded = self.decoded(d4)
		return decoded

class denoisingAEy:
	def __init__(self,bs=16):
		self.bs = bs
	def buildModel(self,hyp = None):
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.model = ConvAutoencoder(hyp).to(self.device)

	def train(self,xTrain,n_epochs = 70):
		trainloader = DataLoader(xTrain, shuffle=True, batch_size=self.bs)
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		scheduler = StepLR(optimizer, step_size=30, gamma=0.4)
		#criterion = SoftDTW(use_cuda, gamma=0.01)
		criterion = nn.MSELoss()
		# specify loss function
		histTrainLoss = []
		# number of epochs to train the model
		for epoch in range(1, n_epochs + 1):
			# monitor training loss
			train_loss = 0.0
			for i, (dataIn, dataOut,idxTrain) in enumerate(trainloader):
				sensor_dataRec = []
				for In in dataIn:
					sensor_dataRec.append(In.to(device=self.device, dtype=torch.float))
				sensor_data = dataOut.to(device=self.device, dtype=torch.float)
				# clear the gradients of all optimized variables
				optimizer.zero_grad()
				# forward pass: compute predicted outputs by passing inputs to the model
				outputs = self.model(sensor_dataRec)
				# calculate the loss
				# loss = criterion(outputs[:,0,:,:], sensor_data[:,0,:,:])
				loss = criterion(outputs, sensor_data)
				loss = criterion(np.squeeze(outputs), np.squeeze(sensor_data))
				
				# backward pass: compute gradient of the loss with respect to model parameters
				loss.mean().backward()
				# loss.backward()
				# perform a single optimization step (parameter update)
				optimizer.step()
				# update running training loss
				train_loss += loss.mean().item()
			
			# print avg training statistics
			scheduler.step()
			train_loss = train_loss / len(trainloader)
			histTrainLoss.append(train_loss)
		return histTrainLoss
	
	def save(self,savePath):
		with open(savePath,'w') as s:
			pickle.dump(self.model,s, protocol=pickle.HIGHEST_PROTOCOL)
	
	def loadModel(self, filePath):
		with open(filePath, 'rb') as m:
			self.model = pickle.load(m)

	def predict(self,xTest):
		testloader =  DataLoader(xTest, shuffle=False, batch_size=1)
		first = True
		y_all = []
		idx_allTest = []
		with torch.no_grad():
			for i, (dataInTest, dataOutTest, label_i,idxTest) in enumerate(testloader):
				y = label_i.cpu().data.numpy()[0]
				testRec = []
				for InTest in dataInTest:
					testRec.append(InTest.to(device=self.device, dtype=torch.float))
				testGT = dataOutTest.to(device=self.device, dtype=torch.float)
				pred = self.model(testRec)
				# pred, testRec, testGT = pred.cpu().data.numpy()[0][0], testRec.cpu().data.numpy()[0][0],testGT.cpu().data.numpy()[0][0]
				pred, testGT, testRec = pred.cpu().data.numpy()[0], testGT.cpu().data.numpy()[0], \
				                        testRec[0].cpu().data.numpy()[0]


				pred_ = deepcopy(testGT)
				idx = idxTest.cpu().data.numpy()[0]
				pred_[0,idx,:] = pred[0,idx,:]
				if first:
					pred_all = pred_
					testGT_all = testGT
					testRec_all = testRec
					y_all.append(y)
					idx_allTest.append(idx)
					first = False
				else:
					pred_all = np.concatenate([pred_all, pred_], axis=0)
					testGT_all = np.concatenate([testGT_all, testGT], axis=0)
					testRec_all = np.concatenate([testRec_all, testRec], axis=0)
					y_all.append(y)
					idx_allTest.append(idx)
			# pred_all = recontrucao do AE
			# testGT_all = dado original
			# testRec = dado reconstruido com media
			# y_all = Labels
			return pred_all, testGT_all, testRec_all,y_all, idx_allTest
