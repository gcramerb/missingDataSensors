import numpy as np
import keras
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm
import pandas as pd
from keras import backend as K

sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler
from utils import saveAll

K.set_image_data_format('channels_first')


def _stream(inp, n_filters, kernel, n_classes):
	hidden = keras.layers.Conv2D(
		filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
		padding='same')(inp)
	hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
	hidden = keras.layers.Conv2D(
		filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
		padding='same')(hidden)
	hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
	
	hidden = keras.layers.Flatten()(hidden)
	
	activation_dense = 'selu'
	kernel_init_dense = 'glorot_normal'
	n_neurons = 50
	dropout_rate = 0.1
	
	# -------------- second hidden FC layer --------------------------------------------
	if kernel_init_dense == "":
		hidden = keras.layers.Dense(n_neurons)(hidden)
	else:
		hidden = keras.layers.Dense(n_neurons, kernel_initializer=kernel_init_dense)(hidden)
	
	hidden = activation_layer(activation_dense, dropout_rate, hidden)
	
	# -------------- output layer --------------------------------------------
	
	hidden = keras.layers.Dense(n_classes)(hidden)
	out = keras.layers.core.Activation('softmax')(hidden)
	
	return out


# pylint: disable=R0201


def activation_layer(activation, dropout_rate, tensor):
	"""Activation layer"""
	import keras
	if activation == 'selu':
		hidden = keras.layers.core.Activation(activation)(tensor)
		hidden = keras.layers.normalization.BatchNormalization()(hidden)
		hidden = keras.layers.noise.AlphaDropout(dropout_rate)(hidden)
	else:
		hidden = keras.layers.core.Activation(activation)(tensor)
		hidden = keras.layers.normalization.BatchNormalization()(hidden)
		hidden = keras.layers.core.Dropout(dropout_rate)(hidden)
	return hidden


def _kernelmlfusion(n_classes, input_shape, kernel_pool):
	width = (16, 32)
	
	streams_input = keras.layers.Input((input_shape[0], input_shape[1], input_shape[2]))
	streams_models = []
	for i in range(len(kernel_pool)):
		streams_models.append(_stream(streams_input, width, kernel_pool[i], n_classes))
	
	if len(kernel_pool) > 1:
		concat = keras.layers.concatenate(streams_models, axis=-1)
		# hidden = self.activation_layer(activation_concat, dropout_rate, concat)
		# -------------- output layer --------------------------------------------
		hidden = keras.layers.Dense(n_classes)(concat)
		out = keras.layers.core.Activation('softmax')(hidden)
	else:
		out = streams_models[0]
	
	# -------------- model buider  --------------------------------------------
	model = keras.models.Model(inputs=streams_input, outputs=out)
	model.compile(loss='categorical_crossentropy',
	              optimizer='RMSProp',
	              metrics=['accuracy'])
	
	return model


if __name__ == '__main__':
	np.random.seed(12227)
	
	if len(sys.argv) > 1:
		data_input_file = sys.argv[1]
	else:
		data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
		path = data_input_file

	# ---------------------------- model parameters -----------------
	pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
	

	n_folds = 14
	for fold_i in range(n_folds):
		dataset_name = 'USCHAD.npz'
		DH = dataHandler()
		DH.load_data(dataset_name=dataset_name, sensor_factor='1.1', path=path)
		n_class = len(pd.unique(DH.dataY))
		dataset = dataset_name.split('.')[0]
		DH.splitTrainTest(fold_i = fold_i)
		trainAcc = DH.dataXtrain[0]
		trainGyr = DH.dataXtrain[1]

		train = np.concatenate([trainAcc,trainGyr],axis = -1)
		train = np.expand_dims(train, axis=1)
		y_train = np.array(DH.dataYrawTrain)
		
	
		_model = _kernelmlfusion(n_class, (train.shape[1], train.shape[2], train.shape[3]), pool)
		# _model.summary()
		# exit()
		_model.fit(train, y_train, cm.bs, cm.n_ep, verbose=0,
		           callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
		           validation_data=(train, y_train))
		
		_model.save('models/USCHAD_ACC_Gyr_fold_' + str(fold_i)+ '.h5')