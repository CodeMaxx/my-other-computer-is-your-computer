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
		return model.predict(testData)

	def cprint(self, str):
		print("\n"+"-"*25+" "+str+" "+"-"*25+"\n")

	def checkAccuracy(self,X_test,model,labels):
		if(model=="lr"):
			mdl = self.Models.lr
		elif(model=="svc"):
			mdl = self.Models.svc
		elif(model=="nn"):
			mdl = self.Models.nn
		elif(model=="knn"):
			mdl = self.Models.knn
		elif(model=="xgbc"):
			mdl = self.Models.xgbc
		elif(model=="rfc"):
			mdl = self.Models.rfc
		elif(model=="vc"):
			mdl = self.Models.vc
		else:
			print(model)
		print(model)
		predictions = self.predict(X_test,mdl)
		print(predictions)
		print(labels)
		correct = sum([predictions[i]==labels[i] for i in range(len(predictions))])
		return correct*100/len(predictions)

def main():
	mode = int(sys.argv[1])
	if(mode==1):
		filename = int(sys.argv[2])
	print("Starting Testing Experiment...")
	print("Extracting Features from test Data...")

	if mode==1:
		p = Preprocessing(fileName)
		X_test = p.get_processed_data(1)
	else:
		p = Preprocessing(1)
		X_test, y_test = p.get_processed_data(0)

	print("Feature extraction complete")

	predictor = Predictor("finalModels.pkl")

	# print(predictor.Models.final_features)
	# print(X_test.columns)

	for feature in predictor.Models.final_features:
		if(feature not in X_test.columns):
			X_test[feature] = 0
			# print(X_test.columns)
	for feature in X_test.columns:
		if(feature not in predictor.Models.final_features):
			del X_test[feature]
			# print(X_test.columns)

	print(X_test.columns)
	X_test.columns = [str(x) for x in X_test.columns]
	X_test = X_test.reindex_axis(sorted(X_test.columns), axis=1)

	print("Normalising...")
	print(X_test)

	# scaled_X_test = np.array(X_test)
	# print(scaled_X_test)
	scaled_X_test = predictor.Models.scaler.transform(X_test.values)
	X_test = pd.DataFrame(scaled_X_test, index = X_test.index, columns = X_test.columns)

	print("Data points normalised")

	modelNames = ["Logistic Regression","SVC","Neural Network","KNN","XGBoost","Random Forest","Voting Classifier"]
	models = ["lr","svc","nn","knn","xgbc","rfc","vc"]

	if mode==1:
		predictions = [predictor.predict(X_test,model) for model in models]
		modelWisePrediction = dict(zip(modelNames,predictions))

	else:
		for i,model in enumerate(models):
			accuracy = predictor.checkAccuracy(X_test,model,p.actualLabels)
			print("Accuracy - "+modelNames[i]+" : ", accuracy)

	print('All Done!')


if __name__ == "__main__":
	main()



