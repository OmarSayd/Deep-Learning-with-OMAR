# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:24:47 2020

@author: SELF ORGANIZED MAP by OMAR for finding frauds in credit card approval
"""

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
markers = ['o', 's']
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
frauds = np.concatenate((mappings[(2,5)], mappings[3,8], mappings[(4,3)]),
                        axis = 0)
frauds = sc.inverse_transform(frauds)








