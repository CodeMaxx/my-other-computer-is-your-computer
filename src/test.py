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


# Pandas for easier data representation
import pandas as pd
import numpy as np
import sys

# Reading and Writing pickle files
from sklearn.externals import joblib

from train import SupervisedModels
from preprocessing import Preprocessing

# Predictor class for a predictor object
class Predictor():

	# Loads the SupervisedModel file fr
	def __init__(self,fileName):
		self.Models = joblib.load(fileName)

	def predict(self,testData,model):
		return self.model.predict(testData)

	def cprint(self, str):
        print("\n" + "-"*25 + " " + str + " " + "-"*25 + "\n")

	def checkAccuracy(self,X_test,model,labels)
		predictions = self.predict(X_test,model)
		correct = sum([predictions[i]==labels[i] for i in range(len(predictions))])
		return correct*100/len(predictions)

def main():
	mode = int(sys.argv[1])
	filename = int(sys.argv[2])
	print("Starting Testing Experiment...")
	print("Extracting Features from test Data...")

	if mode==0:
		p = Preprocessing(fileName)
		X_test = p.get_processed_data(1)
	else:
		p = Preprocessing(1)
		X_test = p.get_processed_data(0)

	print("Feature extraction complete")

	print("Normalising...")

	X_test = p.scaler(X_test)

	print("Data points normalised")

	predictors = Predictor("finalModels.pkl")

	modelNames = ["Logistic Regression","SVC","Neural Network","KNN","XGBoost","Random Forest","Voting Classifier"]
	models = ["lr","svc","nn","knn","xgbc","rf","vc"]

	if mode==0:
		predictions = [predictors.predict(X_test,model) for model in models]
		modelWisePrediction = dict(zip(modelNames,predictions))

	else:
		file = open("../updatedTestLabels.csv",'r')
		labels = [lines[0] for lines in [line.split(',') for line in file.readlines()[1:]]]
		for i,model in models:
			accuracy = predictors.checkAccuracy(X_test,model,labels)
			print("Accuracy - "+modelNames[i]+" : ", accuracy)

	print('All Done!')


if __name__ == "__main__":
	main()



