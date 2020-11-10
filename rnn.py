import numpy as np
import pandas as pd
import math
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class RNN(object):
    def __init__(self):
        self = keras.Sequential()
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.window_size = 14

    def train(self):
        self.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], 7)))
        self.add(Dropout(0.2))

        self.add(LSTM(units=60, activation='relu', return_sequences=True))
        self.add(Dropout(0.2))

        self.add(LSTM(units=80, activation='relu', return_sequences=True))
        self.add(Dropout(0.2))

        self.add(LSTM(units=120, activation='relu'))
        self.add(Dropout(0.2))

        self.add(Dense(units=1))

    def fit(self):
        self.compile(optimizer='adam', loss='mean_squared_error')
        self.fit(self.X_train, self.y_train, epochs=50, batch_size=32)

    def pred(self):
        y_pred = model.predict(X_test)



