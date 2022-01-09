import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

sys.path.insert(0, '../')

from model import DCNNclassifier


# import geomloss


from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from collections import OrderedDict



class clfDCNN(LightningModule):
	def __init__(self, nEpoch=200,input_shape = (1,500,6)):
		super().__init__()
		self.nEpoch = nEpoch
		self.reg = 1e-3
		
		self.model = DCNNclassifier(input_shape)
		self.model.build()
		self.save_hyperparameters()
		self.loss = torch.nn.CrossEntropyLoss()
	
	def forward(self, X):
		return self.model(X)
	
	def training_step(self, batch, batch_idx):
		# opt = self.optimizers()
		data, label = batch['data'], batch['label'].long()
		pred = self.model(data)
		loss = self.loss(pred, label)
		tqdm_dict = {"train_loss": loss.detach()}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
		return output
	
	# def training_epoch_end(self, output):
	# 	metrics = {}
	# 	opt = [i['log'] for i in output]
	#
	# 	keys_ = opt[0].keys()
	# 	for k in keys_:
	# 		metrics[k] = torch.mean(torch.stack([i[k] for i in opt]))
	# 	for k, v in metrics.items():
	# 		self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	#
	def validation_step(self, batch, batch_idx):
		res = self._shared_eval_step(batch, batch_idx)
		metrics = res['log']
		return metrics
	
	def save_params(self,save_path,file):
		path = os.path.join(save_path,file)
		torch.save(self.model.state_dict(), path)
	def load_params(self,save_path,file):
		PATH = os.path.join(save_path, file)
		self.model.load_state_dict(torch.load(PATH))

		# for param in self.model.parameters():
		# 	param.requires_grad =

	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			if k == 'acc':
				metrics['val_' + k] = np.mean(val)
			else:
				metrics['val_' + k] = torch.mean(torch.stack(val))
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		
		data, label = batch['data'], batch['label'].long()
		pred = self.model(data)
		yhat = np.argmax(pred.detach().cpu(), axis=1)
		ytrue = label.cpu().numpy()
		
		acc = accuracy_score(ytrue, yhat)
		f1 = f1_score(ytrue, yhat,average = 'macro')

		
		self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return {'test_acc':acc,'test_f1':f1, 'cm': cm}
	def myTest(self,dl_test):

		yhat= []
		ytrue= []

		
		for i,batch in enumerate(dl_test):
			data, label = batch['data'], batch['label'].long()
			pred = self.model(data)
			yhat.append(np.argmax(pred.detach().cpu(), axis=1))
			ytrue.append(label.cpu().numpy())
		
		yhat = np.concatenate(yhat,axis = 0)
		ytrue = np.concatenate(ytrue,axis = 0)
		acc = accuracy_score(ytrue, yhat)
		f1 = f1_score(ytrue, yhat,average = 'macro')
		cm = confusion_matrix(ytrue, yhat)
		return {'acc':acc,'f1':f1,'cm': cm}

	def _shared_eval_step(self, batch, batch_idx):
		data, label = batch['data'], batch['label'].long()
		pred = self.model(data)
		
		loss = self.loss(pred, label)

		metrics = {"loss": loss}
		
		tqdm_dict = metrics
		result = {
			'progress_bar': tqdm_dict,
			'log': tqdm_dict
		}
		return result
	
	def configure_optimizers(self):
		optimizer = optim.RMSprop(self.model.parameters(),
		                          lr=0.0007,
		                          alpha =0.85,
		                          momentum=0.0,
		                          eps=1e-06,
		                          centered=True,
		                          weight_decay=self.reg)
		#clipnorm=1,
		return optimizer
	

# self.model = load_model(filepath)