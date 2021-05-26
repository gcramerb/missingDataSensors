import numpy as np
import sys
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler
from utils import classesNames


def plotSample():
	data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
	path = data_input_file
	DH = dataHandler()
	DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.1', path=path)
	DH.apply_missing(missing_factor='0.3', missing_sensor='1.0')
	DH.impute('mean')
	DH.splitTrainTest(fold_i=0)
	sample = 500
	labels = classesNames('USCHAD.npz')
	xMiss = DH.dataXmissingTest[0][sample,:,:]
	xtrue = DH.dataXtest[0][sample,:,:]
	y = DH.dataYtest[sample]
	lab = labels[y]

	missing = True
	
	if missing:
		data = pd.DataFrame(xMiss[:, 0], columns=["acc eixo x"])
		data["acc  eixo y"] = xMiss[:, 1]
		data["acc eixo z"] = xMiss[:, 2]
		idx = DH.get_missing_indices()['test'][sample]
		meanx = DH.dataXreconstructedTest[0][sample, idx, 0]
		meany = DH.dataXreconstructedTest[0][sample, idx, 1]
		meanz = DH.dataXreconstructedTest[0][sample, idx, 2]
		data.index = range(0,500)
		data['MissingData X'] = np.nan
		data.iloc[idx,3] = meanx
		data['MissingData Y'] = np.nan
		data.iloc[idx, 4] = meany
		data['MissingData Z'] = np.nan
		data.iloc[idx, 5] = meanz
	else:
		data = pd.DataFrame(xtrue[:, 0], columns=["eixo x"])
		data["eixo y"] = xtrue[:, 1]
		data["eixo z"] = xtrue[:, 2]
	ax = data.plot(style=['g-','y-','b-','r--','r--','r--'],legend = True)
	#ax = data.plot(style=['g-', 'y-', 'b-'])
	#sns.lineplot(data=data, palette="tab10", linewidth=2.5)
	plt.title(f'Activity: {lab}')
	ax.set_ylabel('Aceleração (g)')
	#ax.set_ylabel('dps')
	ax.set_xlabel('Amostras (timestep)')
	plt.savefig('MissingFinal')
plotSample()

a = 1
def rmse_plot():
	data = {0.2: {"mean": "0.20694186", "default": "0.5539755", "last_value": "0.27910498", "frequency": "1.8004411"},
	 0.5: {"mean": "0.21051186", "default": "0.5551676", "last_value": "0.27930617", "frequency": "0.29173514"},
	 0.7: {"mean": "0.21153286", "default": "0.55502284", "last_value": "0.27665937", "frequency": "0.40023482"},
	 0.9: {"mean": "0.21787073", "default": "0.555074", "last_value": "0.27689937", "frequency": "0.51064557"}}
	a = pd.DataFrame(data).T
	a2 = a.iloc[:,0:3]
	a2 = a2.astype('float')
	a2.plot()
	plt.savefig('rmse_plot.png')
	
	
rmse_plot()