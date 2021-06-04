import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from support.early_stopping import EarlyStopping
import numpy as np

import time
import os
import utils
import models
from models import rits_i, brits_i, rits, brits, gru_d, m_rnn
import argparse
import data_loader
import pandas as pd
import ujson as json

from sklearn import metrics

from ipdb import set_trace
import pickle

from dataGeneratorForBRITS import dataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='brits')
parser.add_argument('--hid_size', type=int, default=50)
parser.add_argument('--impute_weight', type=float, default=0.3)
parser.add_argument('--label_weight', type=float, default=1.0)
parser.add_argument('--missingRate',type=str,default= '0.5')

parser.add_argument('--inPath',type=str,default= None)
parser.add_argument('--outPath',type=str,default=None)
parser.add_argument('--dataset',type=str,default="USCHAD.npz")

args = parser.parse_args()

if args.outPath is None:
	args.outPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Datasets\\USCHAD_forBRITS\\')



def train(model, early_stopping,dataPath):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	
	# data_iter = data_loader.get_loader(batch_size=args.batch_size)
	data_iter = data_loader.get_train_loader(dataPath,batch_size=args.batch_size)
	
	for epoch in range(args.epochs):
		model.train()
		
		run_loss = 0.0
		
		for idx, data in enumerate(data_iter):
			data = utils.to_var(data)
			ret = model.run_on_batch(data, optimizer, epoch)
			
			run_loss += ret['loss'].item()
			
			print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(
				epoch, (idx + 1) * 100.0 / len(data_iter),
				       run_loss / (idx + 1.0)))
		
		test_data_iter = data_loader.get_test_loader(
			batch_size=args.batch_size)
		valid_loss = evaluate(model, test_data_iter)
		
		# early stop
		early_stopping(valid_loss, model)
		
		if early_stopping.early_stop:
			print("Early stopping")
			break


def evaluate(model, val_iter):
	model.eval()
	
	labels = []
	preds = []
	
	evals = []
	imputations = []
	
	save_impute = []
	save_label = []
	
	eval_all = []
	imputation_all = []
	
	for idx, data in enumerate(val_iter):
		data = utils.to_var(data)
		ret = model.run_on_batch(data, None)
		
		# print('*********')
		# print('evals:{}'.format(ret['evals'].size()))
		# print('imputations:{}'.format(ret['imputations'].size()))
		
		# save the imputation results which is used to test the improvement of traditional methods with imputed values
		save_impute.append(ret['imputations'].data.cpu().numpy())
		save_label.append(ret['labels'].data.cpu().numpy())
		
		pred = ret['predictions'].data.cpu().numpy()
		label = ret['labels'].data.cpu().numpy()
		is_train = ret['is_train'].data.cpu().numpy()
		
		# only calculate test data
		
		eval_masks = ret['eval_masks'].data.cpu().numpy()
		eval_ = ret['evals'].data.cpu().numpy()
		imputation = ret['imputations'].data.cpu().numpy()
		
		evals += eval_[np.where(eval_masks == 1)].tolist()
		imputations += imputation[np.where(eval_masks == 1)].tolist()
		
		# for dtw error
		eval_all.append(eval_)
		imputation_all.append(imputation)
		
		# evals += eval_[np.where(eval_masks == 1)
		#                and np.where(is_train == 0)].tolist()
		# imputations += imputation[np.where(eval_masks == 1)
		#                           and np.where(is_train == 0)].tolist()
		
		# collect test label & prediction
		pred = pred[np.where(is_train == 0)]
		label = label[np.where(is_train == 0)]
		
		labels += label.tolist()
		preds += pred.tolist()
	
	# labels = np.asarray(labels).astype('int32')
	# preds = np.asarray(preds)
	
	# print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))
	
	# dtw error
	
	loss_dtw = []
	temp_eval = np.concatenate(eval_all, axis=0)
	temp_imputation = np.concatenate(imputation_all, axis=0)
	
	for j, k in zip(temp_eval, temp_imputation):
		loss_dtw.append(dtw(j, k))
	
	evals = np.asarray(evals)
	imputations = np.asarray(imputations)
	
	print('MAE', np.abs(evals - imputations).mean())
	print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
	print('RMSE', sqrt(metrics.mean_squared_error(evals, imputations)))
	print('TDI', np.asarray(loss_dtw).mean())
	
	save_impute = np.concatenate(save_impute, axis=0)
	save_label = np.concatenate(save_label, axis=0)
	pathResult = os.path.relpath('result/')
	np.save(os.path.join(pathResult, '{}_data'.format(args.model)), save_impute)
	np.save(os.path.join(pathResult, '{}_label'.format(args.model)), save_label)
	
	return sqrt(metrics.mean_squared_error(evals, imputations))


def test(model, savepath,dataPath):
	model.load_state_dict(torch.load(savepath))
	
	test_data_iter = data_loader.get_test_loader(dataPath,
		batch_size=args.batch_size)
	valid_loss = evaluate(model, test_data_iter)


def run(dataPath):
	model = getattr(models,
	                args.model).Model(args.hid_size, args.impute_weight,
	                                  args.label_weight)
	total_params = sum(p.numel() for p in model.parameters()
	                   if p.requires_grad)
	print('Total params is {}'.format(total_params))
	
	if torch.cuda.is_available():
		model = model.cuda()
	
	# Early Stopping
	# initialize the early_stopping object
	# early stopping patience; how long to wait after last time validation loss improved.
	patience = 10
	early_stopping = EarlyStopping(savepath=os.path.relpath('result/resultFinalES.pt'), patience=patience, verbose=True)
	train(model, early_stopping,dataPath)
	with open('model_trained.pkl', 'wb') as output:
		pickle.dump(model,output)
	


def evaluate_model(dataPath):
	model = getattr(models,
	                args.model).Model(args.hid_size, args.impute_weight,
	                                  args.label_weight)
	total_params = sum(p.numel() for p in model.parameters()
	                   if p.requires_grad)
	print('Total params is {}'.format(total_params))
	
	if torch.cuda.is_available():
		model = model.cuda()
	
	savepath = os.path.relpath('result/resultFinal.pt')
	test(model, savepath,dataPath)


if __name__ == '__main__':
	#process the data:
	
	i = 0
	#DG = dataGenerator(missing = args.missingRate)
	#DG.setPath(args.inPath,args.outPath)
	#DG.myPreprocess(fold = i )
	fileName  = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{i}'
	run(os.path.join(args.outPath, fileName+'_train'))
	# evaluate the best model
	evaluate_model(os.path.join(args.outPath, fileName+'_test'))
	