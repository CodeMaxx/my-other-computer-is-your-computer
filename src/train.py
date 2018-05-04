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

from plot import *
from voting import VotingClassifier

########################################################################

class SupervisedModels():
    def __init__(self, X, y):
        self.raw_X = X                           # Data with all features
        self.all_features = list(X.columns)              # List of all features
        # print(self.all_features)
        self.final_features = None               # List of important features
        self.X = None                           # final data with important features
        self.y = y                              # Training point labels
        self.SCORE_IMPORTANCE_THRESHOLD = 0.01


    # train random forest classifier and choose important features based on score
    def feature_selection(self,SCORE_IMPORTANCE_THRESHOLD):
        self.SCORE_IMPORTANCE_THRESHOLD = SCORE_IMPORTANCE_THRESHOLD
        self.feature_selection_core()


    # train random forest classifier and choose important features based on score
    def feature_selection_core(self):
        print("Started Random Forest Feature Selection...")
        # min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # max_leaf_nodes = list(range(5,100,5))
        min_samples_leaf_vals = [1, 2, 3, 4]
        max_leaf_nodes = list(range(5,10,5))
        param_grid = dict(min_samples_leaf=min_samples_leaf_vals,max_leaf_nodes=max_leaf_nodes)

        # Classifier
        _rfc = RandomForestClassifier(
            max_depth=20, random_state=3, n_estimators=100, n_jobs=-1)
        rfc = GridSearchCV(_rfc, param_grid, cv=4, scoring='accuracy')

        # Fitting data
        rfc.fit(self.raw_X, self.y)
        # print(self.all_features, rfc.best_estimator_.feature_importances_)
        feature_scores = dict(
            zip(self.all_features, rfc.best_estimator_.feature_importances_))
        self.final_features = [feature for (feature,score) in feature_scores.items() if score > self.SCORE_IMPORTANCE_THRESHOLD]
        # print(self.raw_X)
        # print(self.final_features)
        fi = np.array(self.final_features, dtype=object)
        self.X = self.raw_X[fi]
        print(len(self.all_features),len(self.final_features))
        print("Feature Selection Complete")


    def train_all(self):
        # For different threshold for feature selection
        # for SCORE_IMPORTANCE_THRESHOLD in list(range(10))/20:
        # self.feature_selection(SCORE_IMPORTANCE_THRESHOLD)
        self.feature_selection(0.001)
        self.trainSVC()
        self.trainXGBC()
        self.trainLogisticRegression()
        self.trainKNN()
        self.train_RandomForest()
        self.trainNeuralNetwork()
        self.trainVotingClassifier()

    def trainSVC(self):
        # Hyperparameter values
        print("SVC training started...")
        C_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        gamma_vals = [0.001, 0.01, 0.1, 1]

        param_grid = dict(C=C_vals, gamma=gamma_vals)

        # Classifier
        _svc = SVC(random_state=42)
        svc = GridSearchCV(_svc, param_grid, cv=4, scoring='accuracy')

        # Fitting data
        svc.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(svc, 'svc.pkl')

        self.svc = svc

        print("SVC trained!")

        # Hyper parameter plot 
        plot_SVC(self.svc, C_vals, gamma_vals, self.SCORE_IMPORTANCE_THRESHOLD)

    def trainXGBC(self):
        print("Starting XGBoost Classifier training...")
        max_depth = list(range(5,15,2))
        learning_rate = list(range(1,10,2))
        learning_rate = [x/10 for x in learning_rate]
        param_grid = dict(max_depth=max_depth,learning_rate=learning_rate)

        _xgbc = XGBClassifier(random_state=42)
        xgbc = GridSearchCV(_xgbc, param_grid, cv=4, scoring='accuracy')

        xgbc.fit(self.X, self.y)
        joblib.dump(xgbc, 'xgbc.pkl')

        self.xgbc = xgbc

        print("XGBoost Classifier trained!")

         # Hyper parameter plot 
        plot_XGBC(self.xgbc, max_depth, learning_rate, self.SCORE_IMPORTANCE_THRESHOLD)


    def trainLogisticRegression(self):
        print("Starting Logistic Regression...")
        C_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        param_grid = dict(C = C_vals)

        _lr = LogisticRegression(random_state=42)
        lr = GridSearchCV(_lr, param_grid, cv=4, scoring='accuracy')

        lr.fit(self.X, self.y)
        joblib.dump(lr, 'lr.pkl')

        self.lr = lr
        print("LR Classifer trained!")

         # Hyper parameter plot 
        plot_LogisticRegression(self.lr, C_vals, self.SCORE_IMPORTANCE_THRESHOLD)


    def trainKNN(self):
        print("Starting KNN training...")
        n_neighbors = list(range(1,20,2)) 
        param_grid = dict(n_neighbors = n_neighbors)

        _knn = KNeighborsClassifier()
        knn = GridSearchCV(_knn, param_grid, cv=4, scoring='accuracy')

        knn.fit(self.X, self.y)
        joblib.dump(knn, 'knn.pkl')

        self.knn = knn
        print("KNN Classifer trained!")

         # Hyper parameter plot 
        plot_KNN(self.knn, n_neighbors, self.SCORE_IMPORTANCE_THRESHOLD)


    def train_RandomForest(self):
        print("Starting Random Forest training...")
        # Hyperparameter values
        min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_leaf_nodes = list(range(5,100,5))
        param_grid = dict(min_samples_leaf=min_samples_leaf_vals,max_leaf_nodes=max_leaf_nodes)

        # Classifier
        _rfc = RandomForestClassifier(
            max_depth=20, random_state=3, n_estimators=100, n_jobs=-1)
        rfc = GridSearchCV(_rfc, param_grid, cv=4, scoring='accuracy')

        # Fitting data
        rfc.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(rfc, 'random_forest.pkl')

        self.rfc = rfc
        print("Random Forest Classifier trained!")

        # Plot for hyper parameter tuning for give threshold
        plot_RandomForest(self.rfc, min_samples_leaf_vals,  max_leaf_nodes, self.SCORE_IMPORTANCE_THRESHOLD)
    
    def trainNeuralNetwork(self):
        print("Starting Neural Network training...")
        # Hyperparameter values
        hidden_layer_sizes_vals = [
            (8, 4), (8, 8), (16, 8), (16, 16), (32, 16), (32, 32), (64, 64), (96, 96)]
        max_iter_vals = [5, 10, 20, 50, 100, 200]

        param_grid = dict(
            hidden_layer_sizes=hidden_layer_sizes_vals, max_iter=max_iter_vals)

        # Classifier
        _nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(64, 2), random_state=1)
        nn = GridSearchCV(_nn, param_grid, cv=4, scoring='accuracy')

        # Fitting data
        nn.fit(self.X, self.y)
        # Create pickle file for model
        joblib.dump(nn, 'neural_network.pkl')
        self.nn = nn
        print("Neural Network trained!")

        # Plot for hyper parameter tuning for give threshold
        plot_MLPClassifier(self.nn, hidden_layer_sizes_vals, max_iter_vals, self.SCORE_IMPORTANCE_THRESHOLD)

    def trainVotingClassifier(self):
        print("Starting Voting Classifier...")

        # Initialising Voting Classifier
        VC = VotingClassifier([self.svc,self.xgbc,self.lr,self.knn,self.rfc,self.nn])
        self.VC = VC
        # Create pickle file for model
        joblib.dump(self.VC, 'voting_classifier.pkl')
        print("Voting Classifier trained!")
 


class SemiSupervisedModels():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def trainSemiSupervised(self):
        raise NotImplementedError
