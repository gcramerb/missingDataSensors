
"""
Here we are going to Implement some standatd Methods for Time Series reconstruction.
These ones are: ARIMA,SARIMA, MatrixFactorization (MF) Multiple imputation by Chained Equations(MICE) and
Expectation Maximization (EM).

"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
from  statsmodels.imputation.mice import MICE
from fancyimpute import IterativeImputer



import pandas as pd
import numpy as np
from copy import deepcopy


def Sarimax(xMissing,missingAxis,seasonal = True):
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
			model_fit = model.fit(disp=False)
			idx_missing = np.argwhere(np.isnan(data))  # All axis has the same missing points
			idx_missing = idx_missing.flatten()
			data[idx_missing] = model_fit.predict(start=idx_missing[0], end=idx_missing[-1], dynamic=True)
	return xMissing

def MICE(xMssing):
	#model = IterativeImputer()
	#xMissing = model.fit_transform(xMissing)
	return xMissing

	
			
