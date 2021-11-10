""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import seaborn as sns

diabetes = load_diabetes()
data_train, data_test, target_train, target_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)
lr = LinearRegression()
lr.fit(X=data_train, y=target_train)
coef = lr.coef_
intercept = lr.intercept_
predicted = lr.predict(data_test)
expected = target_test
print(predicted[:20])
print(expected[:20])

# how many sameples and How many features?
# 442 samples and 10 features
# print(diabetes.data.shape)


# What does feature s6 represent?
# s6 is a target feature that represents the glu, blood sugar level
# print(diabetes.DESCR)


# print out the coefficient
print(coef)


# print out the intercept
print(intercept)


# create a scatterplot with regression line
plt.plot(expected, predicted, ".")

x = np.linsspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
