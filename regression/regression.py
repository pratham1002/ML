""" Usage: python regression.py train.csv test.csv """

import pandas as pd
import numpy as np
import sys

""" Training the data model """

# reading the csv into a pandas data set
train_data = pd.read_csv(sys.argv[1])

# converts the pandas dataset to a numpy ndarray
train = np.array(train_data)

# getting the values of x, y from the training dataset
train_x = train[:, 0]
train_y = train[:, 1]

# average values of x, y in the training dataset
av_train_x = np.mean(train_x)
av_train_y = np.mean(train_y)

# the numerator and denominator summations for the linear regression formula
numerator_sum = np.sum((train_x - av_train_x) * (train_y - av_train_y))
denominator_sum = np.sum((train_x - av_train_x) ** 2)

# the value slope given by linear regression formula
slope = numerator_sum / denominator_sum

# the regression line must pass through the averge values of both x, y
intercept = av_train_y - slope * av_train_x

# the regression line
print('y =', slope, '* x +', intercept)

""" Testing the data model """

# reading the csv into a pandas data set
test_data = pd.read_csv(sys.argv[2])

# converts the pandas dataset to a numpy ndarray
test = np.array(test_data)

# getting the values of x, y from the test dataset
test_x = test[:, 0]
test_y = test[:, 1]     # actual input data

# getting the expected values of y from the regression model
model_y = [(slope * x + intercept) for x in test_x]            # data calculated by model

# variance of model obtained from calcution
model_av = np.mean(model_y)
model_variance = np.mean((model_y - model_av) ** 2)

# variance of test dataset
test_av = np.mean(test_y)
test_variance = np.mean((test_y - test_av) ** 2)

# the accuracy of the model is given by r_square
r_square = model_variance / test_variance

print('Accuracy =', r_square)

