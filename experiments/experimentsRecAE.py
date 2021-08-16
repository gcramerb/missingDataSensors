from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Autoencoder.convAE import denoisingAE
import sys
import os
sys.path.insert(0, "../")
from utils.dataHandler import dataHandler
from Autoencoder.modelUtils.custom_losses import SoftDTW,DCT_Loss,W_MSE

## getting the data:
m = '0.5'
DH = dataHandler()
DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.1.0')
DH.apply_missing(missing_factor=m, missing_sensor='1.0')
DH.impute('mean')
DH.splitTrainTest()
#train, trainRec, testRec = DH.get_data_keras()
#train_data, test_data = DH.get_data_pytorch()
train_data ,test_data = DH.get_data_pytorch(index=True)
trainloader = DataLoader(train_data, shuffle=True, batch_size=16)
testloader = DataLoader(test_data, shuffle=False, batch_size=1)

criterion = SoftDTW(use_cuda,gamma=0.01)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# number of epochs to train the model
n_epochs = 70

myModel = pytorchModel()
myModel.train(n_epoch,trainloader,optimizer)

tag = f'expUSCHAD_{j}'
path = '../../resultadosPytorch/wmse/'
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, path)
pred, testGT, testRec,y_all = myModel.predict(testloader)

#metrics = eval_result(pred_all,testGT_all)
#json.dump()
#print(metrics)
