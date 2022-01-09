import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as MSE
import sys, os, json
from math import log10,sqrt

class absoluteMetrics:
	def __init__(self,xTrue,xRec,idx=None):
		if idx is None:
			idx = []
			for i in range(xTrue.shape[0]):
				diff = (xRec[i,:,0] - xTrue[i,:,0])**2
				aux = np.where(diff != .0)[0]
				idx.append(np.array(list(range(aux[0],aux[-1]+1))))
			idx = np.array(idx)
		axis = xTrue.shape[-1]
		self.dataOri = np.zeros([idx.shape[0],idx.shape[1],axis])
		self.dataRec =  np.zeros([idx.shape[0],idx.shape[1],axis])
		for i in range(xTrue.shape[0]):
			self.dataOri[i,:,:] = xTrue[i,idx[i],:]
			self.dataRec[i,:,:] = xRec[i,idx[i],:]

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
				mse = np.mean(self.dataOri[i,:, j]- self.dataRec[i,:, j])**2
				
				if mse == 0:  # MSE is zero means no noise is present in the signal .
					# Therefore PSNR have no importance.
					psnr_v = 20 * log10(0.1)
				else:
					psnr_v = 20 * log10(max_pixel / sqrt(mse))
				psnr.append(psnr_v)
			psnrMean.append(np.mean(psnr))
			psnr = []
		return np.mean(psnrMean)

	def myMSE(self):
		mse_list = []
		axis = self.dataOri.shape[-1]
		for k in range(3):
			mse = np.square(np.subtract(self.dataOri[:,:,k], self.dataRec[:,:,k])).mean(axis=1)
			mse_list.append(mse.mean())
		return np.mean(mse_list)
	def MAPE(self):
		resp = []
		axis = self.dataOri.shape[-1]
		for k in range(axis):
			d = np.subtract(self.dataOri[:,:,k], self.dataRec[:,:,k])/self.dataOri[:,:,k]
			r = np.abs(d).mean(axis=1)
			resp.append(r.mean())
		return np.mean(resp)
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
		result['MAPE'] = self.MAPE()
		result['PSNR'] = self.psnr()
		a,b,c = self.pearson_corr()
		result['corrX'] = a
		result['corrY'] = b
		result['corrZ'] = c
		return result



	def summarizeMetric(resList):
		"""
		resList: list of dictionaries
		"""
		resp = dict()
		mse = [i['MSE'] for i in resList]
		mape = [i['MAPE'] for i in resList]
		icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
		resp['MSE_down'] = icMse[0]
		resp['MSE_up'] = icMse[1]
		resp['MSE'] = np.mean(mse)
		
		icMape = st.t.interval(alpha=0.95, df=len(mape) - 1, loc=np.mean(mape), scale=st.sem(mape))
		resp['MAPE_down'] = icMape[0]
		resp['MAPE_up'] = icMape[1]
		resp['MAPE'] = np.mean(mape)
		
		corrX = [i['corrX'] for i in resList]
		corrY = [i['corrY'] for i in resList]
		corrZ = [i['corrZ'] for i in resList]
		resp['corrX'] = np.mean(corrX)
		resp['corrY'] = np.mean(corrY)
		resp['corrZ'] = np.mean(corrZ)
		resp['MSE_list'] = mse
		resp['corr_list'] = [np.mean([a, b, c]) for a, b, c in zip(corrX, corrY, corrZ)]
		return resp
	
	def summarizeMetric(resList):
		"""
		resList: list of dictionaries
		"""
		resp = dict()
		mse = [i['MSE'] for i in resList]
		icMse = st.t.interval(alpha=0.95, df=len(mse) - 1, loc=np.mean(mse), scale=st.sem(mse))
		resp['MSE_down'] = icMse[0]
		resp['MSE_up'] = icMse[1]
		resp['MSE'] = np.mean(mse)
		corrX = [i['corrX'] for i in resList]
		corrY = [i['corrY'] for i in resList]
		corrZ = [i['corrZ'] for i in resList]
		resp['corrX'] = np.mean(corrX)
		resp['corrY'] = np.mean(corrY)
		resp['corrZ'] = np.mean(corrZ)
		resp['MSE_list'] = mse
		resp['corr_list'] = [np.mean([a, b, c]) for a, b, c in zip(corrX, corrY, corrZ)]
		
		return resp
	


