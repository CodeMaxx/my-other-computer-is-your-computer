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

# Grid search cross-validation for tuning hyperparameters
from sklearn.model_selection import GridSearchCV

########################################################################

class SupervisedModels():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def trainSVC(self):
        # Hyperparameter values
        C_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        gamma_vals = [0.001, 0.01, 0.1, 1]

        param_grid = dict(C=C_vals, gamma=gamma_vals)

        # Classifier
        _svc = SVC(random_state=42)
        svc = GridSearchCV(_svc, param_grid, cv=5, scoring='accuracy')

        # Fitting data
        svc.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(svc, 'svc.pkl')

        self.svc = svc

    def trainXGBC(self):
        raise NotImplementedError

    def trainLogisticRegression(self):
        raise NotImplementedError

    def trainKNN(self):
        raise NotImplementedError

    def train_RandomForest(self):
        # Hyperparameter values
        min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        param_grid = dict(min_samples_leaf=min_samples_leaf_vals)

        # Classifier
        _rfc = RandomForestClassifier(
            max_depth=20, random_state=3, n_estimators=100, n_jobs=-1)
        rfc = GridSearchCV(_rfc, param_grid, cv=5, scoring='accuracy')

        # Fitting data
        rfc.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(rfc, 'random_forest.pkl')

        self.rfc = rfc

    def trainNeuralNetwork(self):
        raise NotImplementedError


class SemiSupervisedModels():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def trainSemiSupervised(self):
        raise NotImplementedError
