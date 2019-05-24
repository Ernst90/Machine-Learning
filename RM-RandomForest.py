import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
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


# Load teh respective datasets into console 
X_train = np.loadtxt('X_train_reg.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_reg.csv', delimiter=',', skiprows=1)[:, 1]
X_data_test = np.loadtxt('X_test_reg.csv', delimiter=',', skiprows=1)

## Create a pipline to process the tuning of parameters over classifier
pipe = make_pipeline(RandomForestClassifier())
'''Hyper Parameter Tuning
hyperparameters = {'randomforestclassifier__random_state': range(0,100,1),
                    'randomforestclassifier__n_estimators' : range(1,70,1),
                    'randomforestclassifier__max_features' : [None, 'sqrt', 'log2'],
                   'randomforestclassifier__max_depth' : [None, 4, 5, 6, 7, 8, 10]}

# Tune model using cross-validation
clrf = RandomizedSearchCV(pipeline, hyperparameters, n_iter=1000)
'''

clrf = RandomForestClassifier(random_state = 9)

# Fitting model with the hyperamaters
clrf.fit(X_train, y_train)

y_predict = clrf.predict(X_data_test)

# Store answer of prediction in a csv file
test_header = "Id,PRP"
n_points = X_data_test.shape[0]
y_predict_pp = np.ones((n_points, 2))
y_predict_pp[:, 0] = range(n_points)
y_predict_pp[:, 1] = y_predict
np.savetxt('regr_rf_submission.csv', y_predict_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
print(y_predict)




##Accuracy of the RFC model with different random-STATE

features_train, features_test, target_train, target_test = train_test_split(X_train,y_train,random_state = 11)


rfc2 = RandomForestClassifier(random_state = 1)
rfc2.fit(X_train, y_train)
y_pred4 = rfc2.predict(features_test)
# Train and Test Accuracy
print("rfc2 Train Accuracy 2:: ", accuracy_score(y_train, rfc2.predict(X_train)))
print("rfc2 Test Accuracy  2:: ", accuracy_score(target_test, y_pred4))

rfc3 = RandomForestClassifier(random_state = 15)
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
