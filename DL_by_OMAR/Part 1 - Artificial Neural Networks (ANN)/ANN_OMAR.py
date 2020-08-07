# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:57:59 2020

@author: ANN by OMAAR, 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
"""
X = dataset.iloc[:, 3:13].values     #[:, [2,3]] is also correct !!
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

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

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

# Making a Confusion matrix
from sklearn.metrics import confusion_matrix, f1_score, classification_report
cm = confusion_matrix(y_test, y_pred_1)
print('\n')
f1 = f1_score(y_test, y_pred_1)
print('\n')
report = classification_report(y_test, y_pred_1)"""