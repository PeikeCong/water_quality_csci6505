import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocess import *  # Assuming read_data is defined here

class WaterQualityDataset:
    def __init__(self, feature_dataset, target_dataset, sequence_length=16):
        self.feature_dataset = feature_dataset
        self.target_dataset = target_dataset
        self.sequence_length = sequence_length
        self.features, self.targets = self.preprocess_data()
        self.x, self.y = self.create_sequences()

    def preprocess_data(self):
        features = self.feature_dataset.values
        targets = self.target_dataset['Dissolved Oxygen'].values.reshape(-1, 1)  # Select target
        # Normalize targets
        scaler = StandardScaler()
        targets = scaler.fit_transform(targets).flatten()  # Flatten to 1D for ARIMA
        self.target_scaler = scaler  # Save scaler for inverse transformation if needed
        return features, targets

    def create_sequences(self):
        x, y = [], []
        for i in range(self.sequence_length, len(self.targets)):
            x.append(self.features[i - self.sequence_length:i])
            y.append(self.targets[i])
        x = np.array(x)
        y = np.array(y)
        return x, y

    def get_data(self, test_size=0.1, validation_size=0.1):
        total_size = len(self.y)
        train_size = int(total_size * (1 - test_size - validation_size))
        validation_size = int(total_size * validation_size)

        X_train = self.x[:train_size]
        X_val = self.x[train_size:train_size + validation_size]
        X_test = self.x[train_size + validation_size:]

        y_train = self.y[:train_size]
        y_val = self.y[train_size:train_size + validation_size]
        y_test = self.y[train_size + validation_size:]

        return X_train, X_val, X_test, y_train, y_val, y_test

feature_dataset = read_data(PATH_KNN_PCA)
target_dataset = read_data(PATH_KNN)
sequence_length = 16

dataset = WaterQualityDataset(feature_dataset, target_dataset, sequence_length)

X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_data(test_size=0.1, validation_size=0.1)

train_val = np.concatenate([y_train, y_val])

model = auto_arima(train_val, seasonal=False, stepwise=True, suppress_warnings=True)

n_test = len(y_test)
forecast = model.predict(n_periods=n_test)

forecast_original_scale = dataset.target_scaler.inverse_transform(forecast.reshape(-1, 1))

test_mse = mean_squared_error(y_test, forecast_original_scale.flatten())
print(f"Test MSE: {test_mse:.4f}")
