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

###############################
# 1. SVM
# 2. XGBoost
# 3. Logistic Regression
# 4. kNeighbour Classifier
# 5. Random Forest
# 6. Semi-supervised Learning
# 7. Neural Network
###############################


# Pandas for easier data representation
import pandas as pd
import numpy as np

# Reading and Writing pickle files
from sklearn.externals import joblib

# Various Classifiers used
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.semi_supervised import 
from sklearn.neural_network import MLPClassifier

# A classifier which implements voting on the classifiers given to it
class VotingClassifier():

	def __init__(self,listClassifiers):
		self.listClassifiers = listClassifiers

	# Function to predict 
	def predict(self,testData):
		predicted = [x.predict(testData) for x in self.listClassifiers]
		predicted = [[predicted[i][j] for i in range(len(listClassifiers))] for j in range(len(testData))]

		# Getting counts of each predicted value 
		valCountPair = [numpy.unique(a, return_counts=True) for a in predicted]
		counts = [dict(zip(x[0],x[1])) for x in valCountPair]

		# Getting the majority prediction for each data point
		finalPrediction = [max(x,key=x.get) for x in counts]

		return finalPrediction


