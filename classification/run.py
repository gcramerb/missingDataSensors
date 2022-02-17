import sys,argparse
import numpy as np
import scipy.stats as st

sys.path.insert(0, '../')

# import geomloss
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from trainer import clfDCNN
from dataModule import DM


parser = argparse.ArgumentParser()
parser.add_argument('--slurm', action='store_true')
parser.add_argument('--dataset', type=str, default="MHEALTH")
parser.add_argument('--sensor', type=str, default="acc")
args = parser.parse_args()

if args.slurm:
	inPath = '/storage/datasets/sensors/LOSO/'
	outPath = '../saved/'
	verbose = 0
else:
	inPath = None
	outPath = 'C:\\Users\\gcram\\Documents\\GitHub\\missingDataSensors\\saved\\'
	verbose = 1

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), st.sem(a)
	h = se * st.t.ppf((1 + confidence) / 2., n - 1)
	return m, m - h, m + h

filter = {}
filter['acc'] = slice(0,3)
filter['gyr'] = slice(3,6)
filter['accGyr'] = slice(0,6)



def runClassifier():

	result = {}
	result['acc'] = []
	result['f1'] = []
	for fold_i in range(14):
		
		dm = DM(f'{args.dataset}.npz',path = inPath)
		dm._setup(fold_i,filter[args.sensor])
		input_shape = (1, 500, 6) if args.sensor == 'accGyr' else (1,500,3)
		model = clfDCNN(input_shape = input_shape)
		my_logger = None
		
		if False:
			wandb_logger = WandbLogger(project='missingData', log_model='all', name='exploring' + args.file_name)
	
			my_logger.watch(model)
		
		early_stopping = EarlyStopping('val_loss', mode='min', patience=10, verbose=True)
	
		trainer = Trainer(gpus=1,
		                  logger=my_logger,
		                  check_val_every_n_epoch=1,
		                  gradient_clip_val=1,
		                  gradient_clip_algorithm="value",
		                  max_epochs=200,
		                  progress_bar_refresh_rate=verbose,
		                  callbacks=[early_stopping])
		
		trainer.fit(model, datamodule=dm)
		metrics = trainer.test(model,dm.test_dataloader())
		result['acc'].append(metrics[0]['test_acc'])
		result['f1'].append(metrics[0]['test_f1'])
		model.save_params(save_path = outPath,file = f'{args.dataset}_fold_{fold_i}_{args.sensor}')
		del model,trainer,dm
	return {'acc':mean_confidence_interval(result['acc']),'f1':mean_confidence_interval(result['f1'])}
if __name__ == '__main__':
	result  = runClassifier()
	print('Classifier for: ',args.sensor)
	print(result)

