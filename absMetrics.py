import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as MSE
import scipy.stats as st
import sys
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler
from utils import saveAll


if __name__ == '__main__':
	
	# define configurations to calculate the result:
	
	np.random.seed(12227)
	if len(sys.argv) > 1:
		path = sys.argv[1]
		dataset_name = path.split('\\')[-1]
		dataset = dataset_name.split('.')[0]
		impMeth =  sys.argv[2]
		si =  sys.argv[3]  # simple impute
	else:
		path = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\USCHAD.npz'
		dataset_name = path.split('\\')[-1]
		dataset = dataset_name.split('.')[0]
		impMeth = 'DAE_sdtw'
		default_impute = ['mean', 'default', 'last_value']
		impute_list = ['DAE_wmse', 'DAE_mse', 'DAE_sdtw']
		si = False  # simple impute
		
	
	missing_list = [0.5]
	Result = dict()
	for miss in missing_list:
		Result[miss] = dict()
		result_file = dataset + '_' + miss + '_RMSE_' + impMeth.replace("_", "")
		rmse_list = []
		for fold_i in range(14):
			file = dataset + '_' + miss + '_' + imp + str(fold_i) + '.npz'
			DH = dataHandler()
			testRec, testTrue, idxAll = DH.get_reconstructed(dataset_name,miss,imp,si,path,file,fold_i)
			
			# como aplicar vetorizacao nessa parte
			for i in range(len(testRec)):
				x= (testTrue[i,:, 0] - testRec[i, :, 0])**2
				idx = np.where(x != 0)[0]
				rmse_list.append(MSE(testTrue[i, idx, 0], testRec[i, idx, 0], squared=False))
				rmse_list.append(MSE(testTrue[i, idx, 1], testRec[i, idx, 1], squared=False))
				rmse_list.append(MSE(testTrue[i, idx, 2], testRec[i, idx, 2], squared=False))
			del DH
		
			Result[miss][impMeth] = str(np.mean(rmse_list))
			path_result = os.path.realpath('metrics')
			
			result_name = f'{result_file}.json'
			result_file_name = os.path.join(path_result, result_name)
			
			with open(result_file_name, "w") as write_file:
				json.dump(Result, write_file)

	def eval_result(self, xpred, xGT=None):
		def psnr_metric(original, compressed):
			mse = np.mean((original - compressed) ** 2)
			if mse == 0:  # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
				return 100
			max_pixel = 1
			psnr = 20 * log10(max_pixel / sqrt(mse))
			return psnr
		
		if xGT is None:
			xGT = self.dataXtest[0]
		result = dict()
		result['RMSE'] = dict()
		result['RMSE_missing'] = dict()
		result['PSNR'] = dict()
		j = 0
		for axis in ['x', 'y', 'z']:
			rmsePred = []
			rmsePredM = []
			psnr = []
			for i in range(len(xGT)):
				rmsePred.append(mean_squared_error(xGT[i, :, j], xpred[i, :, j], squared=False))
				# rmseRec.append(mean_squared_error(self.dataXtest[0][i, :, j], self.dataXreconstructedTest[0][i, :, j], squared=False))
				idx_missing = np.argwhere(np.isnan(self.dataXmissingTest[0][i, :, j]))
				idx_missing = np.squeeze(idx_missing)
				rmsePredM.append(mean_squared_error(xGT[i, idx_missing, j], xpred[i, idx_missing, j], squared=False))
				# rmseRecM.append(mean_squared_error(self.dataXtest[0][i, idx_missing, j], self.dataXreconstructedTest[0][i, idx_missing, j], squared=False))
				psnr.append(psnr_metric(xGT[i, :, j], xpred[i, :, j]))
			j = j + 1
			result['RMSE']['autoEncoder  ' + axis] = np.mean(rmsePred)
			result['RMSE_missing']['autoEncoder  ' + axis] = np.mean(rmsePredM)
			result['PSNR']['autoEncoder  ' + axis] = np.mean(psnr)
		result['RMSE']['all_axis'] = np.mean(
			[result['RMSE']['autoEncoder  x'], result['RMSE']['autoEncoder  y'], result['RMSE']['autoEncoder  z']])
		result['RMSE_missing']['all_axis'] = np.mean(
			[result['RMSE']['autoEncoder  x'], result['RMSE']['autoEncoder  y'], result['RMSE']['autoEncoder  z']])
		result['PSNR']['all_axis'] = np.mean(
			[result['PSNR']['autoEncoder  x'], result['PSNR']['autoEncoder  y'], result['PSNR']['autoEncoder  z']])
		# result['RMSE']['reconstructed  '+ axis] = np.mean(rmseRec)
		self.evalResult = result
		
		return result