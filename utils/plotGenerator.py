import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import json
import seaborn as sn
import sys
sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\")
from dataHandler import dataHandler
from utils import classesNames

#TODO : organizar essa função toda..

def plot_Rec(pred, testRec, testGT, path, label, tag='',imputeType = 'mean'):
	sensors = ['acc', 'gyr', 'mag']
	axis = [' x', ' y', ' z']
	true = testGT
	rec = testRec
	
	f, axarr = plt.subplots(3, sharex=True, sharey=True)
	# pyplot.figure()
	
	# determine the total number of plots
	# n, off = imgs_B.shape[2] + 1, 0
	# sensor = np.squeeze(acc)
	# plot total TRUE acc
	axarr[0].plot(true[:, 0], color='green', label='eixo x')
	axarr[0].plot(true[:, 1], color='blue', label='eixo y')
	axarr[0].plot(true[:, 2], color='red', label='eixo z')
	axarr[0].set_title('ACC Original - {}'.format(label))

	axarr[0].set_ylabel(f'Aceleração(g)')
	axarr[0].legend()
	# plot total REconstructed acc
	axarr[1].plot(rec[:, 0], color='green', label='eixo x')
	axarr[1].plot(rec[:, 1], color='blue', label='eixo y')
	axarr[1].plot(rec[:, 2], color='red', label='eixo z')
	axarr[1].set_title('ACC ' + imputeType)

	axarr[1].set_ylabel(f'Aceleração(g)')
	axarr[1].legend()
	
	# plot total predction acc
	axarr[2].plot(pred[:, 0], color='green', label='eixo x')
	axarr[2].plot(pred[:, 1], color='blue', label='eixo y')
	axarr[2].plot(pred[:, 2], color='red', label='eixo z')
	axarr[2].set_title('ACC Autoencoder em Y')
	axarr[2].set_xlabel('Amostras (timesteps)')
	axarr[2].set_ylabel(f'Aceleração(g)')
	axarr[2].legend()
	# plt.show()
	
	# plt.savefig(f"C:\\Users\gcram\Documents\Github\TCC\ + folder + '\' {label_file_name}.png")
	file_name = path + f'/{label}_{tag}.png'
	plt.savefig(file_name)
	# plt.savefig("../folder/%s_%s.png" % (label, file_name))
	
	plt.close()
def save_fig(impute, out_file='img1'):
	classifier = 'Sena'
	dataset = 'USCHAD'
	metric = 'F1'
	result = dict()
	for impute_type in impute:
		for file in glob.glob(f"Catal/*.json"):
			with open(file) as json_file:
				data = json.load(json_file)
			missingRate = np.float(file.split('_')[-2])
			try:
				result[impute_type][missingRate] = data[metric]
			except:
				result[impute_type] = dict()
				result[impute_type][missingRate] = data[metric]
	
	df = pd.DataFrame(result)
	df = df.sort_index()
	df = df*100
	# df = df.rename(columns={'value': 'last_value'})
	ax = df.plot(style=['ro-', 'ko-', 'go-', 'bo-','yo-'], title='Missing data impact')
	ax.set_xlabel('Missing data rate (%)')
	ax.set_ylabel(f'{metric} score (%)')
	plt.savefig(f'{out_file}.png')




	
def plot_cm(arqName = ''):
	infos = arqName.split('_')
	dataset = infos[1]
	missingRate = infos[2]
	algo = infos[3].split('.')[0]

	cm = np.load(arqName)['cm']
	cm = cm / cm.astype(np.float).sum(axis=1)
	labels = classesNames('USCHAD.npz')
	lab = [labels[x] for x in range(len(labels))]
	df_cm = pd.DataFrame(cm, index=lab, columns=lab)
	plt.figure(figsize=(10, 7))
	sn.heatmap(df_cm, annot=True,fmt='.2f',cmap='Blues')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title(f'Confusion matrix\n MDR: {missingRate} - recontruction: {algo}')
	plt.show()




def get_result():
	classifier = 'Sena'
	dataset = 'USCHAD'
	metric = 'F1'
	result = dict()
	for file in glob.glob(f"Catal/*.json"):
		with open(file) as json_file:
			data = json.load(json_file)
		infos  = file.split('_')
		missingRate = np.float(infos[2])
		method = infos[3].split('.')[0]
		if len(infos)> 4:
			method = method + '_' + infos[-1].split('.')[0]
		
		try:
			result[method][missingRate] = data[metric]
		except:
			result[method] = dict()
			result[method][missingRate] = data[metric]
	
	df = pd.DataFrame(result)
	df = df.sort_index()
	df = df * 100
	df.index = df.index * 100
	return df


def plot_rec_3():
	fold = 0
	labels = classesNames('USCHAD.npz')
	sample = 500
	miss = '0.5'
	
	data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
	path = data_input_file + f'USCHAD_{miss}_AEY_sdtw{fold}.npz'
	data = np.load(path, allow_pickle=True)
	dataRec = data['deploy_data']
	yRec = data['classes']
	
	DH = dataHandler()
	DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.0', path=data_input_file)
	# DH.apply_missing(missing_factor=miss, missing_sensor='1.0')
	# DH.impute('mean')
	DH.splitTrainTest(fold_i=fold)
	
	xtrue = DH.dataXtest[0][sample, :, :]
	
	y = DH.dataYtest[sample]
	lab = labels[y]
	
	pred = dataRec[sample]
	diff = xtrue[:, 0] - pred[:, 0]
	idx_Notmissing = np.argwhere(diff == .0)
	idx_Notmissing = idx_Notmissing.flatten()
	idx_missing = list(set(range(500)) - set(idx_Notmissing))
	meanX = np.mean(xtrue[idx_Notmissing, 0])
	meanY = np.mean(xtrue[idx_Notmissing, 1])
	meanZ = np.mean(xtrue[idx_Notmissing, 2])
	
	from copy import deepcopy
	xMiss = deepcopy(xtrue)
	xMiss[idx_missing, :] = [meanX, meanY, meanZ]
	a = 1
	
	testRec = xMiss
	testGT = xtrue
	path = './'
	label = lab
	plot_Rec(pred, testRec, testGT, path, label, tag='RecExample_final', imputeType='mean')
	

miss = '0.3'
sample = 5100
data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
DH = dataHandler()
DH.load_data(dataset_name='USCHAD.npz', sensor_factor='1.0', path=data_input_file)
DH.apply_missing(missing_factor=miss, missing_sensor='1.0')

missingIdx = DH.get_missing_indices()[sample]
data = pd.DataFrame(DH.dataXmissing[0][sample,:,:],columns = ['eixo x','eixo y','eixo z'])
aux = np.array([np.nan]*500)
aux[missingIdx] = DH.dataX[0][sample,missingIdx,0]
data['Local Dados ausntes x'] = aux
aux = np.array([np.nan]*500)
aux[missingIdx] = DH.dataX[0][sample,missingIdx,1]
data['Local Dados ausntes y'] = aux
aux = np.array([np.nan]*500)
aux[missingIdx] = DH.dataX[0][sample,missingIdx,2]
data['Local Dados ausntes z'] = aux

plot = data.plot(style=['b-', 'r-', 'g-','b:', 'r:', 'g:'], title='Simulação dados Ausentes',xlabel = 'Amostras (timestep)',ylabel = 'Aceleração (g)')
fig = plot.get_figure()
fig.savefig('images/missingImputation.png')


#plot_rec_3()
#plot_cm('Sena_USCHAD_0.2_AEmse.npz')
#impute=['AEYmse', 'AEYwmse', 'AEYsdtw','mean']
#impute=['DAEmse', 'DAEwmse', 'DAEsdtw','mean']
#save_fig(impute, out_file = 'DAEf.png')

# d = get_result()
# d['dados completos'] = 50.46
#
# #impute=['DAE_mse', 'DAE_wmse', 'DAE_sdtw','dados completos']
# impute=['AEY_mse', 'AEY_wmse', 'AEY_sdtw','dados completos']
# impute=['last_value', 'default','mean','dados completos']
# impute = ['last_value','AEY_mse','DAE_sdtw','dados completos']
#
# ax = d[impute].plot(style=['ro-', 'ko-','bo-', 'go--'], title='Impacto de dados ausentes')
# ax.set_xlabel('Taxa de dados ausentes (%)')
# ax.set_ylabel(f'F1 score (%)')
# ax.set_xlim([0,90])
# ax.set_ylim([35,55])
#
# plt.savefig('AEY_final')

# df = df.rename(columns={'value': 'last_value'})
# ax = df.plot(style=['ro-', 'ko-', 'go-', 'bo-', 'yo-'], title='Missing data impact')
# ax.set_xlabel('Missing data rate (%)')
# ax.set_ylabel(f'{metric} score (%)')
# plt.savefig(f'{out_file}.png')


def plot_result(self, pred, path, sample=0, tag='teste'):
	sensors = ['acc', 'gyr', 'mag']
	axis = [' x', ' y', ' z']
	true = self.dataXtest[0][sample]
	rec = self.dataXreconstructedTest[0][sample]
	
	s = self.dataYtest[sample]
	label = self.labelsNames[s]
	
	f, axarr = plt.subplots(3, sharex=True, sharey=True)
	# pyplot.figure()
	
	# determine the total number of plots
	# n, off = imgs_B.shape[2] + 1, 0
	# sensor = np.squeeze(acc)
	# plot total TRUE acc
	axarr[0].plot(true[:, 0], color='green', label='x')
	axarr[0].plot(true[:, 1], color='blue', label='y')
	axarr[0].plot(true[:, 2], color='red', label='z')
	axarr[0].set_title('ACC Original - {}'.format(label))
	axarr[0].legend()
	# plot total REconstructed acc
	axarr[1].plot(rec[:, 0], color='green', label='x')
	axarr[1].plot(rec[:, 1], color='blue', label='y')
	axarr[1].plot(rec[:, 2], color='red', label='z')
	axarr[1].set_title('ACC ' + self.imputeType)
	axarr[1].legend()
	
	# plot total predction acc
	axarr[2].plot(pred[:, 0], color='green', label='x')
	axarr[2].plot(pred[:, 1], color='blue', label='y')
	axarr[2].plot(pred[:, 2], color='red', label='z')
	axarr[2].set_title('ACC Autoencoder ')
	axarr[2].legend()
	# plt.show()
	
	# plt.savefig(f"C:\\Users\gcram\Documents\Github\TCC\ + folder + '\' {label_file_name}.png")
	file_name = path + f'/{label}_{tag}.png'
	plt.savefig(file_name)
	# plt.savefig("../folder/%s_%s.png" % (label, file_name))
	
	plt.close()
