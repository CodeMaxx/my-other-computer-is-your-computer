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
        self.raw_X = X                           # Data with all features
        self.all_features = list(X)              # List of all features
        self.final_features = None               # List of important features
        self.X = None                           # final data with important features
        self.y = y                              # Training point labels
        # self.SCORE_IMPORTANCE_THRESHOLD = 0.1


    # train random forest classifier and choose important features based on score
    def feature_selection(SCORE_IMPORTANCE_THRESHOLD):
        #
        min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_leaf_nodes = list(range(5,100,5))
        param_grid = dict(min_samples_leaf=min_samples_leaf_vals,max_leaf_nodes=max_leaf_nodes)

        # Classifier
        _rfc = RandomForestClassifier(
            max_depth=20, random_state=3, n_estimators=100, n_jobs=-1)
        rfc = GridSearchCV(_rfc, param_grid, cv=5, scoring='accuracy')

        # Fitting data
        rfc.fit(self.raw_X, self.y)

        feature_scores =  dict(zip(self.all_features, rfc.feature_importances_)) 
        self.final_features = [feature for (feature,score) in feature_scores.items() if score > SCORE_IMPORTANCE_THRESHOLD]
        self.X = self.raw_X[self.final_features]


    def train_all(self):
        # For different threshold for feature selection
        for SCORE_IMPORTANCE_THRESHOLD in list(range(10))/20:
            self.feature_selection(SCORE_IMPORTANCE_THRESHOLD)
            self.trainSVC()
            self.trainXGBC()
            self.trainLogisticRegression()
            self.trainKNN()
            self.train_RandomForest()
            self.trainNeuralNetwork()

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
        max_depth = list(range(2,20))
        learning_rate = list(range(1,10))
        learning_rate = [x/10 for x in learning_rate]
        param_grid = dict(max_depth=max_depth,learning_rate=learning_rate)

        _xgbc = XGBClassifier(random_state=42)
        xgbc = GridSearchCV(_xgbc, param_grid, cv=5, scoring='accuracy')

        xgbc.fit(self.X, self.y)
        joblib.dump(xgbc, 'xgbc.pkl')

        self.xgbc = xgbc

    def trainLogisticRegression(self):
        param_grid = dict()

        _lr = LogisticRegression(random_state=42)
        lr = GridSearchCV(_lr, param_grid, cv=5, scoring='accuracy')

        lr.fit(self.X, self.y)
        joblib.dump(lr, 'lr.pkl')

        self.lr = lr

    def trainKNN(self):
        n_neighbors = list(range(1,20)) 
        param_grid = dict(n_neighbors = n_neighbors)

        _knn = KNeighborsClassifier(random_state=42)
        knn = GridSearchCV(_knn, param_grid, cv=5, scoring='accuracy')

        knn.fit(self.X, self.y)
        joblib.dump(knn, 'knn.pkl')

        self.knn = knn

    def train_RandomForest(self):
        # Hyperparameter values
        min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_leaf_nodes = list(range(5,100,5))
        param_grid = dict(min_samples_leaf=min_samples_leaf_vals,max_leaf_nodes=max_leaf_nodes)

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
        # Hyperparameter values
        hidden_layer_sizes_vals = [
            (8, 4), (8, 8), (16, 8), (16, 16), (32, 16), (32, 32), (64, 64), (96, 96)]
        max_iter_vals = [5, 10, 20, 50, 100, 200]

        param_grid = dict(
            hidden_layer_sizes=hidden_layer_sizes_vals, max_iter=max_iter_vals)

        # Classifier
        _nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(64, 2), random_state=1)
        nn = GridSearchCV(_nn, param_grid, cv=5, scoring='accuracy')

        # Fitting data
        nn.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(nn, 'model4.pkl')
        self.nn = nn


class SemiSupervisedModels():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def trainSemiSupervised(self):
        raise NotImplementedError
