# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:20:48 2020

@author: GOOGLE STOCK PRICE from 2012-2016 (5y) by OMAR
"""

                                   #PART 1, Data Preprocessing 
# Importing the labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scalling, this time we use normalization rather than standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 Timesteps with 1 output
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping, must see Keras documentation > recurrent layers > 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                                     
                                   #PART 2, Building the RNN
# Importing the keras library and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding first LSTM layer and dropout regularization  
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

# Adding second LSTM layer and dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding third LSTM layer and dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding fourth LSTM layer and dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

# fitting the regressor into training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

                                   #PART 3, making prediction and visualization
# getting the real stock price for 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
                                   
# getting the predicted stock price for 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price for JAN 2017')
plt.plot(predicted_stock_price, color = 'yellow', label = 'Predicted Google Price for JAN 2017')
plt.xlabel('JAN 2017, excluding saturday and sunday')
plt.ylabel('Google Stock Price')
plt.legend(bbox_to_anchor=(1,0.5))
plt.show()




                                   
                                   