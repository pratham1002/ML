import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
av_train_x = sum(train_x) / len(train_x)
av_train_y = sum(train_y) / len(train_y)

# the numerator and denominator summations for the linear regression formula
numerator_sum = 0
denominator_sum = 0

# iterating over x, y values to compute the numerator and denominator
for x, y in zip(train_x, train_y):
    numerator_sum += (x - av_train_x) * (y - av_train_y)
    denominator_sum += (x - av_train_x) ** 2

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

model_y = []            # data calculated by model

# getting the expected values of y from the regression model
for x in test_x:
    model_y.append(slope * x + intercept)

# variance of model obtained from calcution
model_av = sum(model_y) / len(model_y)
model_variance = sum((y - model_av) ** 2 for y in model_y) / len(model_y) 

# variance of test dataset
test_av = sum(test_y) / len(test_y)
test_variance = sum((y - test_av) ** 2 for y in test_y) / len(test_y)

# the accuracy of the model is given by r_square
r_square = model_variance / test_variance

print(r_square)

