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

# Reading and Writing pickle files
from sklearn.externals import joblib

from train import SupervisedModels
from preprocessing import Preprocessing

# Predictor class for a predictor object
class predictor():

	# Loads the SupervisedModel file fr
	def __init__(self,fileName):
		self.Models = joblib.load(fileName)

	def predict(self,testData,model):
		return self.model.predict(testData)


def main():
	print("Starting Testing Experiment...")
	print("Extracting Features from test Data...")
	p = Preprocessing(test)

	X_test = p.get_processed_data()

	print("Feature extraction complete")

	print("Normalising...")

	X_train = p.scaler(X_train)

	print("Data points normalised")

	print("Training Classifiers...")

	models = SupervisedModels(X_train, y_train)
	models.train_all()


	# Create a pickle file for model
	joblib.dump(models,"finalModels.pkl")

	print("Trained All Models")

	print("Average CV accuracy - Logistic Regression: ", models.lr.best_score_)
	print("Average CV accuracy - SVC: ", models.svc.best_score_)
	print("Average CV accuracy - Neural Network: ", models.nn.best_score_)
	print("Average CV accuracy - KNN: ", models.knn.best_score_)
	print("Average CV accuracy - XGBoost: ", models.xgbc.best_score_)
	print("Average CV accuracy - Random Forest: ", models.rfc.best_score_)

	print('All Done!')


if __name__ == "__main__":
	main()



