#! /usr/bin/env python3

#  Copyright(c) Akash Trehan
#  All rights reserved.

#  This code is licensed under the MIT License.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files(the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

# All imports
from pyparsing import Word, hexnums, WordEnd, Optional, alphas, alphanums
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import pickle
import concurrent.futures
import array
from copy import copy
# from train import SupervisedModels, SemiSupervisedModels
import logging
# Reading and Writing pickle files
from sklearn.externals import joblib

# Grid search cross-validation for tuning hyperparameters
from sklearn.model_selection import GridSearchCV

from train import SupervisedModels

#######################################################################

class Preprocessing():
	INSTRN_BIGRAM_THRESHOLD = 20
	BYTE_BIGRAM_THRESHOLD = 100

	def __init__(self, mode):
		self.mode = mode
		if mode==0:
			self.samples_base_dir = '../feature-dump/'
			# self.train_files = list(
			# 	set([i[:20] for i in os.listdir(self.samples_base_dir)]))[:20]
			with open("../updatedTrainLabels.csv") as file:
				for line in file:
					lines = line.split(',')
					self.train_files.append(line[0])
			self.feature_dump = "../feature-dump/"
			self.trainingLabels = "../trainLabels.csv"
			self.targetFeatureDump = "../feature-dump-train/"
		elif mode==1:
			self.samples_base_dir = '../feature_dump/'
			# self.test_files = list(
			# 	set([i[:20] for i in os.listdir(self.samples_base_dir)]))[:2]
			with open("../updatedTestLabels.csv") as file:
				for line in file:
					lines = line.split(',')
					self.train_files.append(line[0])
			self.feature_dump = "../feature_dump/"
			self.targetFeatureDump = "../feature-dump-test/"


	def get_processed_data(self):
		mode = self.mode
		i = 0
		train_data_points_ = pd.DataFrame()
		with concurrent.futures.ProcessPoolExecutor() as executor:
			for features in executor.map(self._extract_features, self.train_files):
				train_data_points_ = pd.concat([train_data_points_, features], axis=0)
				print(i,len(train_data_points_))
				i += 1
			train_data_points_.fillna(0, inplace=True)
			if mode==0:
				train_data_labels_ = self._get_labels()
		if mode==0:
			return (train_data_points_, train_data_labels_)
		else:
			return train_data_points_

	def _getPixelIntensity(self, filename):
		f = open(self.samples_base_dir + filename)
		imageArray = array.array("B")
		imageArray = np.fromfile(f, dtype='B')
		imageArray = imageArray[-1000:]
		f.close()
		return imageArray

	def _extract_features(self, filename):
		# print(filename)
		# use WordEnd to avoid parsing leading a-f of non-hex numbers as a hex
		address_format = Word(hexnums, exact=8) + WordEnd()
		byte_format = Word(hexnums, exact=2) + WordEnd()
		instrn_line_format = ".text:" + address_format + \
			(byte_format*(1,))("bytes") + Word(alphas, alphanums)("instruction")
		byte_line_format = address_format + (byte_format*(1,))("bytes")

		instrn_unigram = defaultdict(int)
		instrn_bigram = defaultdict(int)
		byte_unigram = defaultdict(int)
		byte_bigram = defaultdict(int)
		segments = defaultdict(int)

		# Check if instrunction n-gram and segment size already there
		if(os.path.isfile(self.feature_dump + filename + "_INSTRN_UNIGRAM.pkl")!=True):
			with open(self.samples_base_dir + filename + ".asm", 'r', encoding='Latin-1') as file:
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
							if v > Preprocessing.INSTRN_BIGRAM_THRESHOLD and k[0] != 0})

			joblib.dump(instrn_unigram,self.feature_dump + filename+"_INSTRN_UNIGRAM.pkl")
			joblib.dump(instrn_bigram,self.feature_dump + filename+"_INSTRN_BIGRAM.pkl")
			joblib.dump(segments,self.feature_dump + filename+"_SEGMENT_SIZE.pkl")

		else:
			segments = joblib.load(self.feature_dump + filename+"_SEGMENT_SIZE.pkl")
			instrn_unigram = joblib.load(self.feature_dump + filename+"_INSTRN_UNIGRAM.pkl")
			instrn_bigram = joblib.load(self.feature_dump + filename+"_INSTRN_BIGRAM.pkl")

		# Check if byte n-gram already there
		if(os.path.isfile(self.feature_dump + filename+"_BYTE_UNIGRAM.pkl")!=True):
			with open(self.samples_base_dir + filename + ".bytes", 'r', encoding='Latin-1') as file:
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

			byte_bigram = defaultdict(int, {k:v for k,v in byte_bigram.items() \
											if v > Preprocessing.BYTE_BIGRAM_THRESHOLD and k[0] != 0})

			joblib.dump(byte_unigram,self.feature_dump + filename+"_BYTE_UNIGRAM.pkl")
			joblib.dump(byte_bigram,self.feature_dump + filename+"_BYTE_BIGRAM.pkl")

		else:
			byte_unigram = joblib.load(self.feature_dump + filename+"_BYTE_UNIGRAM.pkl")
			byte_bigram = joblib.load(self.feature_dump + filename+"_BYTE_BIGRAM.pkl")


		# Check if pixel Intensity feature already there
		if(os.path.isfile(self.feature_dump + filename+"_PIXEL_INTENSITY.pkl")!=True):
			pixelIntensity = self._getPixelIntensity(filename + ".asm")
			pixelIntensity = defaultdict(int, {"Pixel" + str(k): pixelIntensity[k] for k in range(1000)})
			joblib.dump(pixelIntensity,self.feature_dump + filename+"_PIXEL_INTENSITY.pkl")
		else:
			pixelIntensity = joblib.load(self.feature_dump + filename+"_PIXEL_INTENSITY.pkl")


		all_features = copy(segments)
		all_features.update(instrn_unigram)
		all_features.update(instrn_bigram)
		all_features.update(byte_unigram)
		all_features.update(byte_bigram)
		all_features.update(pixelIntensity)
		p = pd.DataFrame(all_features, index=[filename,])

		joblib.dump(p,self.targetFeatureDump + filename + "_all_features.pkl")

		print(filename + ' done')
		return p 

	def _get_labels(self):
		trainLabels = pd.read_csv(self.trainingLabels, index_col=0)
		trainLabels = trainLabels['Class']
		trainLabels = trainLabels.loc[self.train_files]
		return trainLabels

	def scaler(self, train_data_):
		scaler = MinMaxScaler()
		scaled_train_data_ = scaler.fit_transform(train_data_)
		scaled_train_data_ = pd.DataFrame(scaled_train_data_, index = train_data_.index, columns = train_data_.columns)
		# print(type(scaled_train_data_))
		return scaled_train_data_


def main():
	print("Starting Experiment...")
	print("Extracting Features...")
	p = Preprocessing(int(sys.argv[1]))

	X_train, y_train = p.get_processed_data()

	# print("Feature extraction complete")

	# print("Normalising...")

	# X_train = p.scaler(X_train)

	# print("Data points normalised")

	# print("Training Classifiers...")

	# models = SupervisedModels(X_train, y_train)
	# models.train_all()


	# # Create a pickle file for model
	# joblib.dump(models,"finalModels.pkl")

	# print("Trained All Models")

	# print("Average CV accuracy - Logistic Regression: ", models.lr.best_score_)
	# print("Average CV accuracy - SVC: ", models.svc.best_score_)
	# print("Average CV accuracy - Neural Network: ", models.nn.best_score_)
	# print("Average CV accuracy - KNN: ", models.knn.best_score_)
	# print("Average CV accuracy - XGBoost: ", models.xgbc.best_score_)
	# print("Average CV accuracy - Random Forest: ", models.rfc.best_score_)

	# print('All Done!')


if __name__ == "__main__":
	main()




