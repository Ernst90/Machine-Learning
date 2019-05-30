#!/usr/bin/env python

import numpy as np
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, mean_squared_error
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict


# Load training and testing data
X_train = np.loadtxt('X_train_reg.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test_reg.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_reg.csv', delimiter=',', skiprows=1)[:, 1]

print('Loaded data file x_train with',len(X_train),'rows.' )
print('Loaded data file x_test with',len(X_test),'rows.' )
print('Loaded data file y_train with',len(y_train),'rows.')
print()

#standardization of data??
#Reg SVM

features_train, features_test, target_train, target_test = train_test_split(X_train,y_train,random_state = 11)

svc2 = SVC()
svc2.fit(X_train, y_train)
y_pred2 = svc2.predict(features_test)
acc_score2 = accuracy_score(target_test, y_pred2)

print('SVM:   ', acc_score2)

#unsure
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predlr = regr.predict(features_test)
scorereg = np.mean(cross_val_score(regr, features_test, target_test))
print("LinReg:  ",scorereg)

#rand reg

rfc2 = RandomForestClassifier(random_state = 18)
rfc2.fit(X_train, y_train)
y_pred4 = rfc2.predict(features_test)
# Train and Test Accuracy
print("rfc2 Train Accuracy 2:: ", accuracy_score(y_train, rfc2.predict(X_train)))
print("rfc2 Test Accuracy  2:: ", accuracy_score(target_test, y_pred4))

rfc3 = RandomForestClassifier(random_state = 19)
rfc3.fit(X_train, y_train)
y_pred3 = rfc3.predict(features_test)
# Train and Test Accuracy
print("rfc3 Train Accuracy 3:: ", accuracy_score(y_train, rfc3.predict(X_train)))
print("rfc3 Test Accuracy  3:: ", accuracy_score(target_test, y_pred3))
print()

rfc4 = RandomForestClassifier(random_state = 5)
rfc4.fit(X_train, y_train)
y_pred4 = rfc4.predict(features_test)
# Train and Test Accuracy
print("rfc4 Train Accuracy 4:: ", accuracy_score(y_train, rfc4.predict(X_train)))
print("rfc4 Test Accuracy  4:: ", accuracy_score(target_test, y_pred4))
scorereg = sqrt(mean_squared_error(target_test, y_pred4))
print("rfc4 root m_s_error1:",scorereg)
scorereg = sqrt(mean_squared_error(y_train, rfc4.predict(X_train)))
print("rfc4 root_m_s_error2:",scorereg)
print()

rfc41 = RandomForestClassifier(random_state = 9)
rfc41.fit(X_train, y_train)
y_pred41 = rfc41.predict(features_test)
print("rfc41 Train Accuracy 41:: ", accuracy_score(y_train, rfc41.predict(X_train)))
print("rfc41 Test Accuracy  41:: ", accuracy_score(target_test, y_pred41))
scorereg = sqrt(mean_squared_error(target_test, y_pred41))
print("rfc41 root m_s_error1:",scorereg)
scorereg = sqrt(mean_squared_error(y_train, rfc41.predict(X_train)))
print("rfc41 root_m_s_error2:",scorereg)
print()

rfc5 = RandomForestClassifier(random_state = 10)
rfc5.fit(X_train, y_train)
y_pred5 = rfc5.predict(features_test)
# Train and Test Accuracy
print("rfc5 Train Accuracy 5:: ", accuracy_score(y_train, rfc5.predict(X_train)))
print("rfc5 Test Accuracy  5:: ", accuracy_score(target_test, y_pred5))
scorereg = sqrt(mean_squared_error(target_test, y_pred5))
print("rfc5 root m_s_error:",scorereg)
scorereg = sqrt(mean_squared_error(y_train, rfc41.predict(X_train)))
print("rfc5 root_m_s_error:",scorereg)
