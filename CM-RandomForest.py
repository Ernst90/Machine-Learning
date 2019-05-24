import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

## Load data into python 
X_train = np.loadtxt('X_train_clas.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train_clas.csv', delimiter=',', skiprows=1)[:, 1]
X_data_test = np.loadtxt('X_test_clas.csv', delimiter=',', skiprows=1)


## setup the parameters for model fitting
''' Optional Hyperparameter tuning
pipeline = make_pipeline(RandomForestClassifier())
# Declare hyperparameters to tune
hyperparameters = {'randomforestclassifier__random_state': range(0,50,1),
                    'randomforestclassifier__n_estimators' : range(60,70,1),
                    'randomforestclassifier__max_features' : [None, 'sqrt', 'log2'],
                   'randomforestclassifier__max_depth' : [None, 4, 5, 6, 7, 8, 10]}
 
# Tune model using cross-validation
#clrf = RandomizedSearchCV(pipeline, hyperparameters, n_iter=1000)
'''

clrf = RandomForestClassifier(max_features='sqrt', n_estimators=8, max_depth=None, random_state=8)

# Fit the model over cross validation 
clrf.fit(X_train, y_train)
y_predict = clrf.predict(X_data_test)


# Store classification data into csv file 
test_header = "Id,EpiOrStroma"
n_points = X_data_test.shape[0]
y_predict_pp = np.ones((n_points, 2))
y_predict_pp[:, 0] = range(n_points)
y_predict_pp[:, 1] = y_predict
np.savetxt('clas_rf_submission.csv', y_predict_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
print(y_predict)



#Print the scores of RF classifier
features_train, features_test, target_train, target_test = train_test_split(X_train,y_train,random_state = 5)

y_predict1 = clrf.predict(features_test)
# Train and Test Accuracy
print("RFC Train Accuracy :: ", accuracy_score(y_train, clrf.predict(X_train)))
print("RFC Test Accuracy  :: ", accuracy_score(target_test, y_predict1))
scorereg = sqrt(mean_squared_error(target_test, y_predict1))
print("Root m_s_error test:",scorereg)
scorereg = sqrt(mean_squared_error(y_train, clrf.predict(X_train)))
print("Root_m_s_error train:",scorereg)


