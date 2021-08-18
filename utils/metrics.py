import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as MSE
import scipy.stats as st
import sys
import pandas as pd
import json
import os
import abc


class absoluteMetrics:
	def __init__(self,xTrue,xRec):
		x = (xTrue[0, :, 0] - xRec[0, :, 0]) ** 2
		idx = np.where(x != 0)[0]
		#TODO : alterar isso pois nao eh generico para qualquer dataset
		if len(idx)%2 != 0:
			idx.append(idx[-1]+1)
		self.dataOri = np.zeros([len(xTrue),len(idx),3])
		self.dataRec =  np.zeros([len(xTrue),len(idx),3])
		for i in range(xTrue.shape[0]):
			x = (xTrue[i, :, 0] - xRec[i, :, 0]) ** 2
			idx = np.where(x != 0)[0]
			if len(idx) % 0 != 0:
				idx.append(idx[-1]+1)
			self.dataOri[i,:,:] = xTrue[i,idx,:]
			self.dataRec[i,:,:] = xRec[i,idx,:]

	def psnr(self):
		"""
		Peak signal to noise ratio
		"""
		shape = self.dataOri.shape
		psnr = []
		psnrMean = []
		max_pixel = 1
		for i in range(shape[0]):
			for j in range(shape[-1]):
				mse = np.mean(self.dataOri[i, self.idx[i], j]- self.dataRec[i, self.idx[i], j])
				
				if mse == 0:  # MSE is zero means no noise is present in the signal .
					# Therefore PSNR have no importance.
					psnr = 20 * log10(0.1)
				else:
					psnr = 20 * log10(max_pixel / sqrt(mse))
				psnr.append(self.psnr_metric())
			psnrMean.append(np.mean(psnr))
			psnr = []
		return np.mean(psnrMean)

	def myMSE(self):
		mse_list = []
		for k in range(3):
			mse = np.square(np.subtract(self.dataOri[:,:,k], self.dataRec[:,:,k])).mean(axis=1)
			mse_list.append(mse.mean())
		return np.mean(mse_list)

	def pearson_corr(self):
		"""
		correlation between two signals!
		"""
		x_corr = np.array([st.pearsonr(x, y)[0] for x, y in zip(self.dataOri[:,:,0],self.dataRec[:,:,0])])
		y_corr = np.array([st.pearsonr(x, y)[0] for x, y in zip(self.dataOri[:,:,1],self.dataRec[:,:,1])])
		z_corr = np.array([st.pearsonr(x, y)[0] for x, y in zip(self.dataOri[:,:,2],self.dataRec[:,:,2])])
		return (x_corr.mean(),y_corr.mean(),z_corr.mean())
	
	def runAll(self):
		result = dict()
		result['MSE'] = self.myMSE()
		result['PSNR'] = self.psnr()
		a,b,c = self.pearson_corr()
		result['corrX'] = a
		result['corrY'] = b
		result['corrZ'] = c
		return result
	
	def summarizeMetric(resList):
		resp = dict()
		mse = [i['MSE'] for i in resList]
		icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
		mse = np.mean(mse)
		resp['MSE'] = mse
		resp['MSE_down'] = icMse[0]
		resp['MSE_up'] = icMse[1]
		resp['PSNR'] = np.mean([i['PSNR'] for i in resList])
		resp['corrX'] = np.mean([i['corrX'] for i in resList])
		resp['corrY'] = np.mean([i['corrY'] for i in resList])
		resp['corrZ'] = np.mean([i['corrZ'] for i in resList])
		return resp
