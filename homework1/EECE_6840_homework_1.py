# EECE 6840 Homework 1
# Created on 9/7/2025
# Created by Mina Gaffney

# Hands on code challenge: Linear Classifier on a Toy Dataset

# Importing things:
import sklearn
import matplotlib.pyplot as plt
import numpy as np

# 1) Generate or load a toy dataset 
# Two features, two classes
# Ex: make two gaussian blobs with sklearn.datasets.make_blobs 

# import and generate test dataset 
from sklearn.datasets import make_blobs
test_points,test_labels = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=0.05, random_state=4)
# using random state = 4 so I can play around with changing the weights on the same
# test data. TODO: remove random_state to randomize test data (if desired)

# plot test dataset and color code based on label
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_labels, cmap='viridis')
plt.title("Test Dataset Blobs")
plt.show()


# 2) Implement predictions: 
# Write a fxn that takes the weights w, bias b, and input x, and returns the sign of w*x+b

# defining weighted sum fxn for perceptron
def percept_weight_sum(X):
    return np.dot(X, weights.reshape(-1, 1)) + bias

# defining activation fxn
# step fxn that is true when the percept_weight_sum is greater than 0
# and false otherwise
def step_fxn(s):
    return np.where(s >= 0, 1, 0)

def error_fxn(y_target, y_hat):
    return y_target.reshape(-1, 1) - y_hat


# 3) Manually adjust parameters 
# adjust (w,b) by hand and see which ones correctly separate the classes 
# plot points and decision line

# chose random small values to initialize weights + bias
# note these weights work for n_samples = 2 & random_state = 4 for make_blobs
# would expect different results for other states/when not setting seed
weights = np.array([-0.1, 0.6])
bias = 1

w_sum = percept_weight_sum(test_points)
print(w_sum) # for debugging

# predict
y_hat = step_fxn(w_sum)
print(y_hat) # for debugging

# calculate error
test_error = error_fxn(test_labels,y_hat)

# plotting decision boundary based on the example shared by the professor
# Decision boundary line: w0 + w1*x + w2*y = 0
x1_min = min(test_points[:,0]) # col 0 = x values
x1_max = max(test_points[:,0])

# I'm guessing w0 is the bias term?
ys = [-(bias + weights[0] * x1_min) / weights[1], -(bias + weights[0] * x1_max) / weights[1]]

test_error_bin = abs(test_error).reshape(-1, 1)
# Map categories to colors
color_map = {0: 'purple', 1: 'green'}
# point_colors = [color_map[cat] for cat in test_error_bin]
#
# # Create custom legend handles
# legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
#                   for color in color_map.values()]
#
# # Create custom legend labels
# legend_labels = ['Correctly Predicted', 'Incorrectly Predicted'] # Adjust based on your categories
# plt.legend(handles=legend_handles, labels=legend_labels, title='Categories')

plt.plot([x1_min, x1_max],ys, label=None)
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_error_bin, cmap = 'PiYG')
plt.title("Test Dataset Blobs")
plt.show()




# 4) automate with a simple loop 
# perceptron update rule for a handful of epochs

# defined several steps above... going to call the pre-defined fxns for this section

# starting by defining learning rate and initializing weights and biases
loop_learning_rate = 0.05
loop_weights = np.random.uniform(low=-0.6, high=0.6, size=(test_labels.shape[0]))
loop_bias = 1
loop_epochs = 10

for i in range(loop_epochs):
    # Calculated weighted sum
    loop_w_sum = percept_weight_sum(test_points)

    # predict:
    y_hat = step_fxn(w_sum)
    print(y_hat)

    # calculate error:
    loop_error = error_fxn(test_labels,y_hat)

    # update:
    loop_weights = loop_weights.reshape(-1, 1) + (loop_learning_rate*loop_error.reshape(-1, 1))
    #loop_bias = loop_bias + (loop_learning_rate*loop_error) #<--- TODO: fix bias term updates

    # plotting decision boundary based on the example shared by the professor
    # Decision boundary line: w0 + w1*x + w2*y = 0
    x1_min = min(test_points[:, 0])  # col 0 = x values
    x1_max = max(test_points[:, 0])

    # I'm guessing w0 is the bias term?
    ys = [-(loop_bias + loop_weights[0] * x1_min) / loop_weights[1], -(loop_bias + loop_weights[0] * x1_max) / loop_weights[1]]

    plt.plot([x1_min, x1_max], ys)
    plt.scatter(test_points[:, 0], test_points[:, 1], c=loop_error, cmap='PiYG')
    plt.title("Test Dataset Blobs")
    plt.show()

test_error = error_fxn(test_labels,y_hat)



# 5) Visualize the results 
# plot the dataset, decision boundary, and highlight misclassified points

# Dataset plotted after generation in section 1
# Decision boundary plotted in section 3


# deliverables: turn in one zip file that combines all parts of the assignment. 
# The zip file should be named homework1.zip and include 
# homework1.docx: answers to all questions; including references in IEEE style
# GenAI transcript.docx: transcript of any LLM convo + code
# README.md
# requirements.txt
# *.py and other necessary files to run your programs

