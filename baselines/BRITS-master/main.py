import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from support.early_stopping import EarlyStopping
import numpy as np

import time,os,sys,argparse
import utils
import models
from models import rits_i, brits_i, rits, brits, gru_d, m_rnn
import data_loader
import pandas as pd
import ujson as json

from math import sqrt
from sklearn import metrics
from tslearn.metrics import dtw, dtw_path

from ipdb import set_trace
import pickle5 as pickle
from dataGeneratorForBRITS import dataGenerator
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error


parser = argparse.ArgumentParser()


parser.add_argument('--slurm', action='store_true')

parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='brits')
parser.add_argument('--hid_size', type=int, default=32)
parser.add_argument('--impute_weight', type=float, default=0.3)
parser.add_argument('--label_weight', type=float, default=1.0)

parser.add_argument('--missingRate',type=str,default= '0.5')
parser.add_argument('--Nfolds',type=int,default= 1)
parser.add_argument('--inPath',type=str,default= os.path.abspath('C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\LOSO\\'))
parser.add_argument('--outPath',type=str,default=None)
parser.add_argument('--dataset',type=str,default="USCHAD.npz")

args = parser.parse_args()

if args.slurm:
	import sys
	sys.path.insert(0, "/home/guilherme.silva/")
	classifiersPath = os.path.abspath('/mnt/users/guilherme.silva/classifiers/trained/')
	from missing_data.metrics import absoluteMetrics
	from classifiers.Catal import Catal
	
	try:
		with open(os.path.join(classifiersPath, f'Catal_USCHAD_{0}.pkl'),'rb') as inp:
			junk = pickle.load(inp)
	except:
		print('Classifiers Not working !!!\n')
	


else:

	args.outPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Smart Sense\\Datasets\\USCHAD_forBRITS\\')
	classifiersPath = os.path.abspath('C:\\Users\\gcram\\Documents\\Smart Sense\\classifiers\\trained\\')
	sys.path.insert(0, 'C:\\Users\\gcram\\Documents\\Github\\')
	from missingDataSensors.utils.metrics import absoluteMetrics
	
	sys.path.insert(0, 'C:\\Users\\gcram\\Documents\\Smart Sense\\')
	from classifiers.Catal import Catal
	


def train(model, early_stopping, dataTrain):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	
	# data_iter = data_loader.get_loader(batch_size=args.batch_size)
	data_iter = data_loader.get_train_loader(dataTrain, batch_size=args.batch_size)
	
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
		
		test_data_iter = data_loader.get_test_loader(dataTrain,
		                                             batch_size=args.batch_size)
		valid_loss, d, b = evaluate(model, test_data_iter)
		
		# early stop
		early_stopping(valid_loss, model)
		
		if early_stopping.early_stop:
			#print("Early stopping")
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
	
	# print('MAE', np.abs(evals - imputations).mean())
	# print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
	# print('RMSE', sqrt(metrics.mean_squared_error(evals, imputations)))
	# print('TDI', np.asarray(loss_dtw).mean())
	
	save_impute = np.concatenate(save_impute, axis=0)
	save_label = np.concatenate(save_label, axis=0)
	pathResult = os.path.relpath('result/')
	return sqrt(mean_squared_error(evals, imputations)), save_impute, save_label


def test(model, savepath, dataTest):
	model.load_state_dict(torch.load(savepath))
	
	test_data_iter = data_loader.get_test_loader(dataTest,
	                                             batch_size=args.batch_size)
	valid_loss, imputed, labels = evaluate(model, test_data_iter)
	return imputed, labels


def run(dataTrain):
	model = getattr(models,
	                args.model).Model(args.hid_size, args.impute_weight,
	                                  args.label_weight)
	total_params = sum(p.numel() for p in model.parameters()
	                   if p.requires_grad)
	# print('Total params is {}'.format(total_params))
	
	if torch.cuda.is_available():
		model = model.cuda()
	
	# Early Stopping
	# initialize the early_stopping object
	# early stopping patience; how long to wait after last time validation loss improved.
	patience = 25
	name = args.dataset.split('.')[0]
	early_stopping = EarlyStopping(savepath=os.path.join(args.outPath, f'model{args.model}_{name}.pt'),
	                               patience=patience, verbose=False)
	train(model, early_stopping, dataTrain)


def evaluate_model(dataTest,saved = False,fold = None):
	if saved:
		with open(os.path.join(classifiersPath,f'model_BRITS_fold{fold}.pkl'),'rb') as inp:
			model = pickle.load(inp)
	else:
		model = getattr(models,
		                args.model).Model(args.hid_size, args.impute_weight,
		                                  args.label_weight)
	total_params = sum(p.numel() for p in model.parameters()
	                   if p.requires_grad)
	#print('Total params is {}'.format(total_params))
	
	if torch.cuda.is_available():
		model = model.cuda()
	
	# savepath = os.path.relpath('result/resultFinalES.pt')
	name = args.dataset.split('.')[0]
	savepath = os.path.join(args.outPath, f'model{args.model}_{name}.pt')
	imputed, labels = test(model, savepath, dataTest)
	return imputed, labels


if __name__ == '__main__':
	# process the data:
	metrics = []
	AM = absoluteMetrics()
	for fold_i in range(args.Nfolds):
		DG = dataGenerator(missing=args.missingRate)
		DG.setPath(args.inPath, args.outPath)
		dataTrain, dataTest,idx_test = DG.myPreprocess(fold=fold_i, save=False)
		fileName = args.dataset.split('.')[0] + '_' + args.missingRate + f'_fold_{fold_i}'
		
		run(dataTrain)
		# evaluate the best model
		# os.path.join(args.outPath, fileName+'_test')
		imp, labels = evaluate_model(dataTest)
		imputed = copy.deepcopy(DG.xTrue)
		for i,idx in enumerate(idx_test):
			imputed[i,idx,0] = imp[i,0,idx]
			imputed[i, idx, 0] = imp[i, 1, idx]
			imputed[i, idx, 0] = imp[i, 2, idx]
		name = os.path.join(args.outPath, fileName)
		
		# np.savez(name,imputed = imputed,labels = labels)

		with open(
				os.path.join(classifiersPath, f'Catal_USCHAD_{fold_i}.pkl'),
				'rb') as inp:
			catal_classifier = pickle.load(inp)
		
		#np.transpose(imputed, (0, 2, 1)
		yPred = catal_classifier.predict(imputed)
		mse = AM.myMSE(DG.xTrue, imputed)
		acc = accuracy_score(np.squeeze(labels),yPred)
		f1 = f1_score(np.squeeze(labels),yPred, average='macro')
		metrics.append([mse, acc, f1])
	
	metricsM = np.mean(metrics, axis=0)
	metrics = np.array(metrics)
	ic_acc = st.t.interval(alpha=0.95, df=len(metrics[:,1]) - 1, loc=np.mean(metrics[:,1]), scale=st.sem(metrics[:,1]))
	ic_f1 = st.t.interval(alpha=0.95, df=len(metrics[:, 2]) - 1, loc=np.mean(metrics[:, 2]),scale=st.sem(metrics[:,2]))
	result = {}
	result['MSE'] = str(metricsM[0])
	result['Acuracy'] = str(metricsM[1])
	result['Acc_icLow'] = ic_acc[0]
	result['Acc_icHigh'] = ic_acc[1]
	result['f1'] = str(metricsM[2])
	result['F1_icLow'] = ic_f1[0]
	result['F1_icHigh'] = ic_f1[1]
	savePath = os.path.join(args.outPath, f'result_{args.dataset.split(".")[0]}_{args.missingRate}')
	with open(savePath + '.json', "w") as write_file:
		json.dump(result, write_file)
