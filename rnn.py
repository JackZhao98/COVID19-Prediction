import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class RNN(object):
    # need in put as state_train_data['California']
    # predicted_feature: 0 for cases, 1 for deaths
    def __init__(self, input_data, predicted_feature):
        self.model = keras.Sequential()
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.window_size = 14
        self.days_predict = 30
        self.training_data = input_data
        self.predicted_feature = predicted_feature

    def preprosess_data(self):
        data_training = np.array(self.training_data)

        for i in range(self.window_size, data_training.shape[0] - self.days_predict):
            self.X_train.append(data_training[i - self.window_size:i])
            self.y_train.append(data_training[i + self.days_predict, self.predicted_feature])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)


    def train(self):
        self.preprosess_data()

        self.model.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], 7)))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=80, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=80, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=120, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=160, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1))

    def fit(self):
        self.train()
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32)

    def predict(self):
        self.fit()
        self.process_test_data()
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def process_test_data(self):
        previous_days = self.training_data.tail(self.window_size + self.days_predict)
        df = previous_days.append([], ignore_index=True)
        inputs = np.array(df)

        for i in range(self.window_size, inputs.shape[0]):
            self.X_test.append(inputs[i - self.window_size:i])
            self.y_test.append(inputs[i, self.predicted_feature])

        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)
