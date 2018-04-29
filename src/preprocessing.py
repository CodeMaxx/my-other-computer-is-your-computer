# All imports
from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from collections import defaultdict
import pandas as pd
from copy import copy
import csv
import pandas as pd 
import numpy as np
import csv
import os
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score
import concurrent.futures
import array

# Grid search cross-validation for tuning hyperparameters
from sklearn.model_selection import GridSearchCV

#######################################################################

address_format = Word(hexnums, exact=8) + WordEnd() # use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
byte_format = Word(hexnums, exact=2) + WordEnd()
instrn_line_format = ".text:" + address_format + (byte_format*(1,))("bytes") + Word(alphas,alphanums)("instruction")
byte_line_format = address_format + (byte_format*(1,))("bytes")


# Globals
SAMPLES_BASE_DIR = '../train/'
TRAIN_FILES = list(set([ i[:len('FNMk3wvliVuQLCe9OTDg')] for i in os.listdir(SAMPLES_BASE_DIR)]))[:3500]

INSTRN_BIGRAM_THRESHOLD = 20
BYTE_BIGRAM_THRESHOLD = 100


def getPixelIntensity(filename):
	file = open(filename+".asm")
	length = os.path.getsize(filename+".asm")
	width = int(length**0.5)
	imageArray = array.array("B")
	imageArray = np.fromfile(file,dtype='B')
	imageArray = imageArray[-1000:]
	file.close()
	return imageArray

def get_features(filename):
	instrn_unigram = defaultdict(int)
	instrn_bigram = defaultdict(int)
	byte_unigram = defaultdict(int)
	byte_bigram = defaultdict(int)
	segments = defaultdict(int)
	with open(SAMPLES_BASE_DIR + filename + ".asm", 'r', encoding='Latin-1') as file:
		prev, now = 0, 0
		for line in file:
			# Filtering lines
			segments[line.split(':')[0]] += 1
			if not line.startswith('.text'):
				continue
			if ' db ' in line or ' dd ' in line or ' dw ' in line or 'align ' in line:
				continue
				
			try:
				result = instrn_line_format.parseString(line)
			except:
				continue
				
			prev = now
			now = result.instruction
			instrn_bigram[(prev, now)] += 1
			instrn_unigram[now] += 1

	instrn_bigram = defaultdict(int, {k:v for k,v in instrn_bigram.items() \
						if v > INSTRN_BIGRAM_THRESHOLD and k[0] != 0})
	with open(SAMPLES_BASE_DIR + filename + ".bytes", 'r', encoding='Latin-1') as file:
	    prev, now = 0, 0
	    for line in file:
	        try:
	            result = byte_line_format.parseString(line)
	        except:
	            continue
			
	        byte_list = list(result.bytes)
	        for byte in byte_list:
	            prev = now
	            now = byte
	            byte_bigram[(prev, now)] += 1
	            byte_unigram[now] += 1

	byte_bigram = defaultdict(int, {k:v for k,v in byte_bigram.items() if v > BYTE_BIGRAM_THRESHOLD and k[0] != 0})

	pixelIntensity = getPixelIntensity(filename)
	pixelIntensity = defaultdict(int, {"Pixel"+str(k):pixelIntensity[k] for k in range(1000)})
	all_features = copy(segments)
	all_features.update(instrn_unigram)
	all_features.update(instrn_bigram)
	all_features.update(byte_unigram)
	all_features.update(byte_bigram)
	all_features.update(pixelIntensity)
	p = pd.DataFrame(all_features, index=[filename,])
	print(filename)
	return p 

def get_labels():
	label = defaultdict(int)
	trainLabels = pd.read_csv("./trainLabels.csv", index_col = 0)
	trainLabels = trainLabels['Class']
	trainLabels = trainLabels.loc[TRAIN_FILES]
	return trainLabels


def get_train_data():
	with concurrent.futures.ProcessPoolExecutor() as executor:
		train_data_points_ = pd.DataFrame()
		for filename, features in zip(TRAIN_FILES, executor.map(get_features, TRAIN_FILES)):
			train_data_points_ = pd.concat([train_data_points_, features], axis=0)
		train_data_points_.fillna(0, inplace=True)
		train_data_labels_ = get_labels()
	return (train_data_points_,train_data_labels_)


def scaler(train_data_):
	scaler = MinMaxScaler()
	scaled_train_data_ = scaler.fit_transform(train_data_)
	return scaled_train_data_


def linearClassifier(train_data_points,train_data_labels):
	c = 1
	penalty = 'l2'
	solver = 'newton-cg'
	multi_class = 'ovr'
	clf = linear_model.LogisticRegression(penalty=penalty,C=c,solver=solver,multi_class=multi_class)
	gsclf = GridSearchCV(clf, {}, cv=2, scoring='accuracy')
	gsclf.fit(train_data_points,train_data_labels)
	return gsclf

print("Starting Experiment...")
print("Extracting Features...")

train_data_points_,train_data_labels_ = get_train_data()

print("Feature extraction complete")


print("Normalising...")
train_data_points_ = scaler(train_data_points_)
print("Data points normalised")
print("Training Classifier...")
model = linearClassifier(train_data_points_,train_data_labels_)
print("Training complete")
print("Dumping model...")
joblib.dump(model, "model.pkl")
print("Average CV accuracy: ", model.best_score_)
