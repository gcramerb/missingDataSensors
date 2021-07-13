
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
	def Sarimax(self,xMissing,missingAxis = [0,1,2],seasonal = True):
		works = True
		if seasonal:
			sOrder = (4,1,1,100)
		else:
			sOrder = (4,1,1,0)
		shp = xMissing.shape
		exog = xMissing[:,:,list(set(range(shp[-1]))-set(missingAxis))]
		for i in range(shp[0]):
			for j in missingAxis:
				data = np.squeeze(xMissing[i,:,j])
				model = SARIMAX(data, exog=exog[i],  seasonal_order=sOrder)
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
				mice_impute = IterativeImputer()
				xMissing[i] = mice_impute.fit_transform(sample)
			except:
				works = False
		return works,xMissing
	
	def MatrixFactorization(self,xMissing):
		works = True
		for i,sample in enumerate(xMissing):
			m = MatrixFactorization()
			try:
				xMissing[i] = m.fit_transform(sample)
			except:
				works = False
		return works,xMissing

	def ExpMaximization(self,xMissing):
		for i,sample in enumerate(xMissing):
			xMissing[i] = impy.em(sample)
		return xMissing
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
	
	def runMethod(self,sMissing,method):
		if method == 'sarimax':
			return self.Sarimax(deepcopy(xMissing))
		elif method == "arx":
			return self.Sarimax(deepcopy(xMissing),seasonal = False)
		elif method == "MICE":
			return self.MICE(deepcopy(xMissing))
		elif method == "matrixFactorization":
			return self.MatrixFactorization(deepcopy(xMissing))
		elif method == "expectationMaximization":
			return self.ExpMaximization(deepcopy(xMissing))
