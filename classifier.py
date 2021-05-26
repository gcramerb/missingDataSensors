"""
Here is a implementation of the classification method propose by:

CAGATAY CATAL. SELIN, T. E. P. G. K. On the use of ensemble of classifiers foraccelerometer-based activity recognition.
Applied Soft Computing,
v. 37, p. 1018–1022,2015

"""


import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import os
import sys
import json
sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler

class Catal:
	def __init__(self):
		self.name = 'Catal'
	
	
	def A(self,sample):
	    feat = []
	    for col in range(0,sample.shape[1]):
	        average = np.average(sample[:,col])
	        feat.append(average)
	
	    return feat
	
	def SD(self,sample):
	    feat = []
	    for col in range(0, sample.shape[1]):
	        std = np.std(sample[:, col])
	        feat.append(std)
	
	    return feat
	
	def AAD(self,sample):
	    feat = []
	    for col in range(0, sample.shape[1]):
	        data = sample[:, col]
	        add = np.mean(np.absolute(data - np.mean(data)))
	        feat.append(add)
	
	    return feat
	
	def ARA(self,sample):
	    #Average Resultant Acceleration[1]:
	    # Average of the square roots of the sum of the values of each axis squared √(xi^2 + yi^2+ zi^2) over the ED
	    feat = []
	    sum_square = 0
	    sample = np.power(sample, 2)
	    for col in range(0, sample.shape[1]):
	        sum_square = sum_square + sample[:, col]
	
	    sample = np.sqrt(sum_square)
	    average = np.average(sample)
	    feat.append(average)
	    return feat
	
	def TBP(self,sample):
	    from scipy import signal
	    feat = []
	    sum_of_time = 0
	    for col in range(0, sample.shape[1]):
	        data = sample[:, col]
	        peaks = signal.find_peaks_cwt(data, np.arange(1,4))
	
	        feat.append(peaks)
	
	    return feat
	
	def feature_extraction(self,X):
	    #Extracts the features, as mentioned by Catal et al. 2015
	    # Average - A,
	    # Standard Deviation - SD,
	    # Average Absolute Difference - AAD,
	    # Average Resultant Acceleration - ARA(1),
	    # Time Between Peaks - TBP
	    X_tmp = []
	    for sample in X:
	        features = self.A(sample)
	        features = np.hstack((features, self.A(sample)))
	        features = np.hstack((features, self.SD(sample)))
	        features = np.hstack((features, self.AAD(sample)))
	        features = np.hstack((features, self.ARA(sample)))
	        #features = np.hstack((features, TBP(sample)))
	        X_tmp.append(features)
	
	    X = np.array(X_tmp)
	    return X
	
	def train_j48(self,X, y):
	    from sklearn import tree
	    clf = tree.DecisionTreeClassifier()
	    #clf = clf.fit(X, y)
	    return clf
	
	def train_mlp(self,X, y):
	    from sklearn.neural_network import MLPClassifier
	    a = int((X.shape[1] + np.amax(y)) / 2 )#Default param of weka, amax(y) gets the number of classes
	    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (a,),
	                        learning_rate_init=0.3, momentum=0.2, max_iter=500, #Default param of weka
	                        )
	    #clf.fit(X, y)
	    return clf
	
	def train_logistic_regression(self,X, y):
	    from sklearn.linear_model import LogisticRegression
	    clf = LogisticRegression(multi_class='ovr')
	    #clf.fit(X, y)
	    return clf
	
	def fit(self,x_train,y_train):
		x_train = self.feature_extraction(x_train)
		self.j_48 = self.train_j48(x_train, y_train)
		self.mlp = self.train_mlp(x_train, y_train)
		self.lr = self.train_logistic_regression(x_train, y_train)
		self.majority_voting = VotingClassifier(estimators=[('dt', self.j_48), ('mlp', self.mlp), ('lr', self.lr)],voting='soft')
		self.majority_voting.fit(x_train, y_train)
	
	def predict(self,test):
		test = self.feature_extraction(test)
		y_pred = self.majority_voting.predict(test)
		return y_pred