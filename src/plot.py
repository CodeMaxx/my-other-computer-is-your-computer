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

# Plotting Graphs
import matplotlib.pyplot as plt


# def plot_RandomForest(rfc):
#     # Hyperparameter values
#     min_samples_leaf_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     # Get all scored returned by GridSearchCV
#     scores = [x[1] for x in rfc.grid_scores_]

#     # Get coordinates of best parameters
#     max_x = rfc.best_params_['min_samples_leaf']
#     max_y = round(rfc.best_score_, 3)

#     # Annotate best point on graph
#     plt.annotate('Max at (%s, %s)' % (str(max_x), str(max_y)), xy=(max_x, max_y), xytext=(max_x, max_y-0.01),
#                  arrowprops=dict(facecolor='red', shrink=0.05))

#     # Plot and label graph
#     plt.plot(min_samples_leaf_vals, scores)
#     plt.xlabel("Min Samples in Leaf - RandomForestClassifier")
#     plt.ylabel("Mean Cross-validation Accuracy")
#     # Save to file
#     plt.savefig("RandomForestClassifier.png")
#     plt.gcf().clear()


# def plot_SVC(svc):
#     # Hyperparameter values
#     C_vals = [0.01, 0.1, 1, 10, 100, 1000, 10000]
#     gamma_vals = [0.001, 0.01, 0.1, 1]
#     scores = [x[1] for x in svc.grid_scores_]
#     scores = np.array(scores).reshape(len(C_vals), len(gamma_vals))
#     for ind, i in enumerate(C_vals):
#         # Plot curves for various values of C
#         plt.plot(gamma_vals, scores[ind], label='C: ' + str(i))

#     # Get coordinates of best parameters
#     max_x = svc.best_params_['gamma']
#     max_C = svc.best_params_['C']
#     max_y = round(svc.best_score_, 3)

#     # Annotate best point on graph
#     plt.annotate('Max at (%s, %s) at C=%s' % (str(max_x), str(max_y), str(max_C)), xy=(max_x, max_y),
#                  xytext=(max_x, max_y-0.2), arrowprops=dict(facecolor='red', shrink=0.05))
#     plt.legend()
#     plt.xscale('log')
#     plt.xlabel("Gamma values - Support Vector Machine Classifier")
#     plt.ylabel("Mean Cross-validation Accuracy")
#     # Save to file
#     plt.savefig("SVC.png")
#     plt.gcf().clear()


# def plot_MLPClassifer(nn):
#     # Hyperparameter values
#     hidden_layer_sizes_vals = [
#         (8, 4), (8, 8), (16, 8), (16, 16), (32, 16), (32, 32), (64, 64), (96, 96)]
#     max_iter_vals = [5, 10, 20, 50, 100, 200]
#     scores = [x[1] for x in nn.grid_scores_]
#     scores = np.array(scores).reshape(
#         len(hidden_layer_sizes_vals), len(max_iter_vals))
#     for ind, i in enumerate(hidden_layer_sizes_vals):
#         # Plot curves for various values of hidden layer
#         plt.plot(max_iter_vals, scores[ind],
#                  label='Hidden Layer Size: ' + str(i))

#     # Get coordinates of best parameters
#     max_x = nn.best_params_['max_iter']
#     max_hidden_layer_sizes = nn.best_params_['hidden_layer_sizes']
#     max_y = round(nn.best_score_, 3)

#     # Annotate best point on graph
#     plt.annotate('Max at (%s, %s) at\nHidden Layers=%s' % (str(max_x), str(max_y), str(max_hidden_layer_sizes)), xy=(max_x, max_y),
#                  xytext=(max_x, max_y-0.08), arrowprops=dict(facecolor='red', shrink=0.05))

#     plt.legend()
#     plt.xscale('log', basex=2)
#     plt.xlabel("Max Iterations - MLPClassifier (Neural Network)")
#     plt.ylabel("Mean Cross-validation Accuracy")
#     # Save to file
#     plt.savefig("MLPClassifier.png")
#     plt.gcf().clear()


# def plot_DecisionTree(dtc):
#     # Hyperparameter values
#     min_samples_split_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#     scores = [x[1] for x in dtc.grid_scores_]

#     # Plot graph
#     plt.plot(min_samples_split_vals, scores)

#     # Get coordinates of best parameters
#     max_x = dtc.best_params_['min_samples_split']
#     max_y = round(dtc.best_score_, 3)

#     # Annotate best point on graph
#     plt.annotate('Max at (%s, %s)' % (str(max_x), str(max_y)), xy=(max_x, max_y), xytext=(max_x, max_y-0.01),
#                  arrowprops=dict(facecolor='red', shrink=0.05))

#     plt.xlabel("Min Samples Split - DecisionTreeClassifier")
#     plt.ylabel("Mean Cross-validation Accuracy")
#     # Save to file
#     plt.savefig("DecisionTreeClassifier.png")
#     plt.gcf().clear()
