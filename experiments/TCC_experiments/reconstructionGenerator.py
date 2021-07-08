from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

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

from convAE_pytorch import ConvAutoencoder
from dataHandler import dataHandler
from utils import saveAll
from ES import  EarlyStopping
from custom_losses import SoftDTW,DCT_Loss,W_MSE
#defining all configurations:
missing_list = [2,3,4,5]
#missing_list = ['0.9']
dataset_name = 'USCHAD.npz'
sensor_factor = '1.1'
bs = 16
lr = 1e-3
loss_list = ['sdtw']


n_epochs = 300
for miss in missing_list:
	for cl in loss_list:
		impute_type = 'AEY_u_' + cl
		dataRec = dict()
		classes = dict()
		deploy = dict()
		for fold_i in range(14):
			DH = dataHandler()
			DH.load_data(dataset_name=dataset_name, sensor_factor=sensor_factor, path=path)
			DH.apply_missing(missing_factor=miss,missing_type = 'u', missing_sensor='1.0')
			DH.impute('RWmean')
			DH.splitTrainTest(fold_i=fold_i)
			train_data, test_data = DH.get_data_pytorch(index = False)
			trainloader = DataLoader(train_data, shuffle=True, batch_size=bs)
			testloader = DataLoader(test_data, shuffle=False, batch_size=1)
			index = DH.get_missing_indices()

			torch.manual_seed(22770)
			use_cuda = torch.cuda.is_available()
			device = torch.device("cuda" if use_cuda else "cpu")

			# initialize the NN
			model = ConvAutoencoder().to(device)
			# specify loss function
			if cl == 'mse':
				criterion = nn.MSELoss()
			elif cl == 'sdtw':
				criterion = SoftDTW(use_cuda,gamma=0.01)
			elif cl == 'dct':
				criterion = DCT_Loss(device)
			elif cl == 'wmse':
				criterion = W_MSE(device)
			else:
				print('ERROR: Loss not defined')
			optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
			es = EarlyStopping(patience=5)
			for epoch in range(1, n_epochs + 1):
				# monitor training loss
				train_loss = 0.0
				for i, (dataIn,dataOut) in enumerate(trainloader):
					# _ stands in for labels, here
					# no need to flatten images
					sensor_dataRec = []
					for In in dataIn:
						sensor_dataRec.append(In.to(device = device, dtype=torch.float))
					sensor_data =dataOut.to(device = device, dtype=torch.float)
					# clear the gradients of all optimized variables
					optimizer.zero_grad()
					# forward pass: compute predicted outputs by passing inputs to the model
					outputs = model(sensor_dataRec)
					# calculate the loss
					if cl == 'sdtw' or cl == 'dct':
						loss = criterion(outputs[:, 0, :, :], sensor_data[:, 0, :, :])
					elif cl == 'wmse':
						loss = criterion(outputs[:, 0, :, :], sensor_data[:, 0, :, :], idx)
					else:
						loss = criterion(outputs, sensor_data)
					# backward pass: compute gradient of the loss with respect to model parameters
					loss.mean().backward()
					#loss.backward()
					# perform a single optimization step (parameter update)
					optimizer.step()
					# update running training loss
					train_loss += loss.mean()
					if es.step(train_loss):
						break  # early stop criterion is met, we can stop now
				scheduler.step()
			first = True
			y_all = []
			with torch.no_grad():
				for i, (dataInTest,dataOutTest,label) in enumerate(testloader):
					label_i = label
					y = label_i.cpu().data.numpy()[0]
					testRec = []
					for InTest in dataInTest:
						testRec.append(InTest.to(device=device, dtype=torch.float))
					testGT = dataOutTest.to(device=device,dtype=torch.float)
					pred = model(testRec)
					#pred, testRec, testGT = pred.cpu().data.numpy()[0][0], testRec.cpu().data.numpy()[0][0],testGT.cpu().data.numpy()[0][0]
					pred, testGT,testRec = pred.cpu().data.numpy()[0], testGT.cpu().data.numpy()[0],testRec[0].cpu().data.numpy()[0]

					if first:
						pred_all = pred
						testGT_all =testGT
						testRec_all = testRec
						deploy_data_all  = testGT
						deploy_data_all[0,index,:] = pred[0,index,:]
						y_all.append(y)
						first = False
					else:
						pred_all = np.concatenate([pred_all,pred],axis = 0)
						testGT_all = np.concatenate([testGT_all, testGT], axis=0)
						testRec_all = np.concatenate([testRec_all, testRec], axis=0)
						deploy_data  = testGT
						deploy_data[0,index,:] = pred[0,index,:]
						deploy_data_all = np.concatenate([deploy_data_all, deploy_data],axis = 0)
						y_all.append(y)
			del DH
			dataset = dataset_name.split('.')[0]
			path_out_name = path + dataset+'_u' + str(miss)\
			                + '_' + impute_type +str(fold_i) +'.npz'
			np.savez(path_out_name,deploy_data = deploy_data_all,classes = y_all)
			print('Saving...',path_out_name)

