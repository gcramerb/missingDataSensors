
"""
Here we are going to Implement some standatd Methods for Time Series reconstruction.
These ones are: ARIMA,SARIMA, MatrixFactorization (MF)
 Multiple imputation by Chained Equations(MICE) and
EM).

"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from  statsmodels.imputation.mice import MICE
from fancyimpute import IterativeImputer,MatrixFactorization
import impyute as impy
import pandas as pd
import numpy as np
from copy import deepcopy

class StandardMethods:
	def __init__(self):
		self.sOrder = (100,0,5,100)
	def bestOrder(self,endog,exog):
		best =  (100,0,5,100)
		min = 999999
		for ar in range(0, 101, 10):
			for int in range(0, 3):
				for ma in range(1, 5):
					for sea in range(0, 251, 50):
						if ar == 0:
							ar = 1
						if sea == 0:
							sea = 2
						aicAll = 0
						for i in range(len(endog)):
							sOrder = (ar, int, ma, sea)
							# Estimate the model with no missing datapoints
							model = SARIMAX(endog[i],exog=exog[i], seasonal_order=sOrder)
							try:
								res = model.fit(disp=False)
								aicAll += res.aic/len(endog)
							except:
								pass
								
						if aicAll < min:
							best = sOrder
							min = aicAll/len(endog)
		return best

	def searchBestOrder(self,dataFull,missingAxis = [0,1,2]):
		resp = []
		shp = dataFull.shape
		exog = dataFull[:,:,list(set(range(shp[-1]))-set(missingAxis))]
		for i in missingAxis:
			endog = dataFull[:,:,i]
			resp.append(self.bestOrder(endog, exog))
		self.sOrder = resp

		self.sOrder= best
	def Sarimax(self,xMissing,missingAxis = [0,1,2]):
		works = True
		shp = xMissing.shape
		exog = xMissing[:,:,list(set(range(shp[-1]))-set(missingAxis))]
		for i in range(shp[0]):
			for j in missingAxis:
				data = np.squeeze(xMissing[i,:,j])
				model = SARIMAX(data, exog=exog[i],  seasonal_order=self.sOrder[j])
				try:
					model_fit = model.fit(disp=False)
					idx_missing = np.argwhere(np.isnan(data))  # All axis has the same missing points
					idx_missing = idx_missing.flatten()
					data[idx_missing] = model_fit.predict(start=idx_missing[0], end=idx_missing[-1], dynamic=True)
				except:
					works = False
		return works, xMissing
	
	def MICE(self,xMissing):
		works = True
		for i,sample in enumerate(xMissing):
			try:
				mice_impute = IterativeImputer(verbose=False)
				xMissing[i] = mice_impute.fit_transform(sample)
			except:
				works = False
		return works,xMissing
	
	def MatrixFactorization(self,xMissing):
		works = True
		for i,sample in enumerate(xMissing):
			m = MatrixFactorization(verbose=False)
			try:
				xMissing[i] = m.fit_transform(sample)
			except:
				works = False
		return works,xMissing

	def ExpMaximization(self,xMissing):
		works = True
		for i,sample in enumerate(xMissing):
			try:
				#xMissing[i] = impy.em(sample)
				xMissing[i] = self.myEM(sample)
			except:
				works =False
		return works,xMissing
	def myEM(self,data):
		loops = 50
		nan_xy = np.argwhere(np.isnan(data))
		for x_i, y_i in nan_xy:
			col = data[:, int(y_i)]
			mu = col[~np.isnan(col)].mean()
			std = col[~np.isnan(col)].std()
			col[x_i] = np.random.normal(loc=mu, scale=std)
			previous, i = 1, 1
			for i in range(loops):
				# Expectation
				mu = col[~np.isnan(col)].mean()
				std = col[~np.isnan(col)].std()
				# Maximization
				col[x_i] = np.random.normal(loc=mu, scale=std)
				# Break out of loop if likelihood doesn't change at least 10%
				# and has run at least 5 times
				delta = (col[x_i] - previous) / previous
				if i > 5 and delta < 0.1:
					data[x_i][y_i] = col[x_i]
					break
				data[x_i][y_i] = col[x_i]
				previous = col[x_i]
		return data
	
	def runAll(self,xMissing):
		results = dict()
		auxWorks,aux = self.Sarimax(deepcopy(xMissing))
		if auxWorks:
			results['SARIMAX'] = aux
		auxWorks, aux = self.Sarimax(deepcopy(xMissing), seasonal=False)
		if auxWorks:
			results['ARX'] = aux
		auxWorks, aux = self.MICE(deepcopy(xMissing))
		if auxWorks:
			results['MICE'] =aux
		auxWorks, aux = self.MatrixFactorization(deepcopy(xMissing))
		if auxWorks:
			results['MF'] =  aux
		auxWorks, aux = self.ExpMaximization(deepcopy(xMissing))
		if auxWorks:
			results['EM'] = aux
		return results
	
	def runMethod(self,xMissing,method):
		if method == 'sarimax':
			return self.Sarimax(deepcopy(xMissing))
		elif method == "arx":
			return self.Sarimax(deepcopy(xMissing))
		elif method == "MICE":
			return self.MICE(deepcopy(xMissing))
		elif method == "matrixFactorization":
			return self.MatrixFactorization(deepcopy(xMissing))
		elif method == "expectationMaximization":
			return self.ExpMaximization(deepcopy(xMissing))
