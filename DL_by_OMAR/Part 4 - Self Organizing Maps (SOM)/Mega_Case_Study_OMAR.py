# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:28:22 2020

@author: Mega Case Study using Hybrid DEEP LEARNING by Omar
"""

# PART 1 : finding frauds with SOM

# importing the labraries
import pandas as pd
import numpy as np

# importing the dataset 
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# building the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15 , learning_rate=0.5, sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', '*']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.7,     
         w[1] + 0.7,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Findings the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,1)], mappings[(6,4)], mappings[(7,7)]),
                        axis = 0)
frauds = sc.inverse_transform(frauds)

# PART 2: Going from unsupervised to supervised deep learning model

# Creating the matrix of feature
customers = dataset.iloc[:, 1:].values

# creating the dependent variable
is_frauds = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_frauds[i] = 1

#feature scaling, because of euclidean distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

# Importing the Keras module with packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer 
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN tom the training set
classifier.fit(customers, is_frauds, batch_size=1, epochs=2)

#Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred),
                        axis = 1) 
y_pred = y_pred[y_pred[:, 1].argsort()]










