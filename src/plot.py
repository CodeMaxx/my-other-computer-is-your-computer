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

# Plotting Graphs
import matplotlib.pyplot as plt


def plot_RandomForest(rfc, min_samples_leaf_vals, max_leaf_nodes, SCORE_IMPORTANCE_THRESHOLD):
    # Hyperparameter values
    # min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Get all scored returned by GridSearchCV
    scores = [x[1] for x in rfc.grid_scores_]

    scores = np.array(scores).reshape(len(min_samples_leaf_vals), len(max_leaf_nodes))
    for ind, i in enumerate(min_samples_leaf_vals):
        # Plot curves for various values of C
        plt.plot(max_leaf_nodes, scores[ind], label='min_samples_leaf: ' + str(i))

    # Get coordinates of best parameters
    max_x = svc.best_params_['max_leaf_nodes']
    max_min_samples_leaf = svc.best_params_['min_samples_leaf']
    max_y = round(svc.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s) at min_samples_leaf=%s' % (str(max_x), str(max_y), str(max_min_samples_leaf)), xy=(max_x, max_y),
                 xytext=(max_x, max_y-0.2), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.legend()
    plt.xlabel("Max Leaf Nodes - RandomForestClassifier")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("RandomForestClassifier"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()

def plot_KNN(knn, n_neighbors, SCORE_IMPORTANCE_THRESHOLD):
    # Hyperparameter values
    # min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Get all scored returned by GridSearchCV
    scores = [x[1] for x in rfc.grid_scores_]

    # Get coordinates of best parameters
    max_x = rfc.best_params_['n_neighbors']
    max_y = round(rfc.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s)' % (str(max_x), str(max_y)), xy=(max_x, max_y), xytext=(max_x, max_y-0.01),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Plot and label graph
    plt.plot(n_neighbors, scores)
    plt.xlabel("Number of Neighbors - K-Means Classifier")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("KNN"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()

def plot_SVC(svc, C_vals, gamma_vals, SCORE_IMPORTANCE_THRESHOLD):
    # Hyperparameter values
    # C_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # gamma_vals = [0.001, 0.01, 0.1, 1]
    scores = [x[1] for x in svc.grid_scores_]
    scores = np.array(scores).reshape(len(C_vals), len(gamma_vals))
    for ind, i in enumerate(C_vals):
        # Plot curves for various values of C
        plt.plot(gamma_vals, scores[ind], label='C: ' + str(i))

    # Get coordinates of best parameters
    max_x = svc.best_params_['gamma']
    max_C = svc.best_params_['C']
    max_y = round(svc.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s) at C=%s' % (str(max_x), str(max_y), str(max_C)), xy=(max_x, max_y),
                 xytext=(max_x, max_y-0.2), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Gamma values - Support Vector Machine Classifier")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("SVC"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()

def plot_LogisticRegression(lr, C_vals, SCORE_IMPORTANCE_THRESHOLD):
    # Hyperparameter values
    # min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Get all scored returned by GridSearchCV
    scores = [x[1] for x in lr.grid_scores_]

    # Get coordinates of best parameters
    max_x = lr.best_params_['C']
    max_y = round(lr.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s)' % (str(max_x), str(max_y)), xy=(max_x, max_y), xytext=(max_x, max_y-0.01),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Plot and label graph
    plt.plot(C_vals, scores)
    plt.legend()
    plt.xscale('log')
    plt.xlabel("C, Regularization Strength - Logistic Regression Classifier")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("LR"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()



def plot_MLPClassifer(nn, hidden_layer_sizes_vals, max_iter_vals, SCORE_IMPORTANCE_THRESHOLD):
    # Hyperparameter values
    # hidden_layer_sizes_vals = [
    #     (8, 4), (8, 8), (16, 8), (16, 16), (32, 16), (32, 32), (64, 64), (96, 96)]
    # max_iter_vals = [5, 10, 20, 50, 100, 200]
    scores = [x[1] for x in nn.grid_scores_]
    scores = np.array(scores).reshape(
        len(hidden_layer_sizes_vals), len(max_iter_vals))
    for ind, i in enumerate(hidden_layer_sizes_vals):
        # Plot curves for various values of hidden layer
        plt.plot(max_iter_vals, scores[ind],
                 label='Hidden Layer Size: ' + str(i))

    # Get coordinates of best parameters
    max_x = nn.best_params_['max_iter']
    max_hidden_layer_sizes = nn.best_params_['hidden_layer_sizes']
    max_y = round(nn.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s) at\nHidden Layers=%s' % (str(max_x), str(max_y), str(max_hidden_layer_sizes)), xy=(max_x, max_y),
                 xytext=(max_x, max_y-0.08), arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()
    # plt.xscale('log', basex=2)
    plt.xlabel("Max Iterations - MLPClassifier (Neural Network)")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("MLPClassifier"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()


def plot_XGBC(xgbc, max_depth, learning_rate, SCORE_IMPORTANCE_THRESHOLD):
    scores = [x[1] for x in xgbc.grid_scores_]

    scores = np.array(scores).reshape(len(max_depth), len(learning_rate))
    for ind, i in enumerate(max_depth):
        # Plot curves for various values of C
        plt.plot(learning_rate, scores[ind], label='max_depth: ' + str(i))

    # Get coordinates of best parameters
    max_x = xgbc.best_params_['learning_rate']
    max_depth = xgbc.best_params_['max_depth']
    max_y = round(xgbc.best_score_, 3)

    # Annotate best point on graph
    plt.annotate('Max at (%s, %s) at max_depth=%s' % (str(max_x), str(max_y), str(max_depth)), xy=(max_x, max_y),
                 xytext=(max_x, max_y-0.2), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.legend()
    plt.xlabel("learning_rate values - XGBoost Classifier")
    plt.ylabel("Mean Cross-validation Accuracy")
    # Save to file
    plt.savefig("XGBC"+str(SCORE_IMPORTANCE_THRESHOLD)+".png")
    plt.gcf().clear()

