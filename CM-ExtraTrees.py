import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

## Load the data into Python 
X_train = np.loadtxt('X_train_clas.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_clas.csv', delimiter=',', skiprows=1)[:, 1]
X_data_test = np.loadtxt('X_test_clas.csv', delimiter=',', skiprows=1)

''' Optional Hyperparameter tuning:
pipeline = make_pipeline(ExtraTreesClassifier())
# Declare hyperparameters to tune
hyperparameters = {'extratreesclassifier__random_state': range(0,50,1),
                    'extratreesclassifier__n_estimators' : range(60,70,1),
                    'extratreesclassifier__max_features' : [None, 'sqrt', 'log2'],
                   'extratreesclassifier__max_depth' : [None, 4, 5, 6, 7, 8, 10]}

# Tune model using cross-validation
#clextr = RandomizedSearchCV(pipeline, hyperparameters, n_iter=1000)

'''
## fitting the model
clextr = ExtraTreesClassifier(random_state = 22)
# Fit the model for the data
clextr.fit(X_train, y_train)
y_predict = clextr.predict(X_data_test)



# store data into the csv file
test_header = "Id,EpiOrStroma"
n_points = X_data_test.shape[0]
y_predict_pp = np.ones((n_points, 2))
y_predict_pp[:, 0] = range(n_points)
y_predict_pp[:, 1] = y_predict
np.savetxt('clas_et_submission.csv', y_predict_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
print(y_predict)


#print the accuracy and score of our data
features_train, features_test, target_train, target_test = train_test_split(X_train,y_train,random_state = 22)

rfenew = RFECV(estimator=clextr, verbose = 0)
rfenew = rfenew.fit(X_train, y_train)
rfenewpred = rfenew.predict(features_test)
print("Extra Trees Accuracy")
# Train and Test Accuracy
print("Train Accuracy :: ", accuracy_score(y_train, rfenew.predict(X_train)))
print("Test Accuracy  :: ", accuracy_score(target_test, rfenewpred))
scorereg = sqrt(mean_squared_error(y_train, rfenew.predict(X_train)))
print("Root_m_s_error train:",scorereg)
scorereg = sqrt(mean_squared_error(target_test, rfenewpred))
print("Root_m_s_error test",scorereg)


