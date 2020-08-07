# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:57:59 2020

@author: ANN by OMAAR, 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:13].values     
y = dataset.iloc[:, 13].values

#encoding categorical feature, independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]   # to avoid dummy variable trap

#spliting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#feature scaling, because of euclidean distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

                                      # PART 2
# Importing the Keras module with packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(rate = 0.1))

# Adding the 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN tom the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

                                       # PART 3
# Making the prediction and importing the model

#Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred_1 = (y_pred > 0.5)

# Predicting with some random parameters
"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: Germany
Credit Score:2 
Gender: Male
Age: 35 years old
Tenure: 0.5 years
Balance: $ 10
Number of Products: 0
Does this customer have a credit card? No
Is this customer an Active Member: Yes
Estimated Salary: $ 150
So, should we say goodbye to that customer?

Solution:
"""
new_pred = classifier.predict(sc_X.transform(np.array([[1, 0, 2, 1, 35, 0.5, 10, 0, 0, 1, 150]])))
new_pred = (new_pred > 0.5)

# Another Example
print(classifier.predict(sc_X.transform(np.array([[1, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) > 0.5)

# Making a Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred_1)

print(classification_report(y_test, y_pred_1))

                                        # PART 4
# EVALUATING, IMPROVING, TUNING the MODEL
# 1_EVALUATING the ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,  batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# 2_IMPROVING the ANN (dropout regularization to reduce the overfitting if needed)

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,31],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator= classifier, param_grid= parameters,
                           scoring= 'accuracy', cv= 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_   # 31, 500, rmsprop
best_accuracy = grid_search.best_score_






