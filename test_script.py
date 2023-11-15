import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import joblib


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index, len(target) - history_size):
        indices = range(i - history_size, i, 1)
        data.append(dataset[indices])
        labels.append(target[i + history_size])  # Adjusted index

    return np.array(data), np.array(labels)


# Get user input for city, step size, and test file
city = input("Enter the city name: ")
step = int(input("Enter the step size: "))
test_file = input("Enter the name of the test file (including path if not in the same directory): ")

# Load the pre-trained model
trained_model = load_model(f'C:/Users/deshik/ml/saved_models/{city}_model_{step}.h5')
scaler_features = joblib.load(f'C:/Users/deshik/ml/saved_models/{city}_scaler_features.pkl')
scaler_target = joblib.load(f'C:/Users/deshik/ml/saved_models/{city}_scaler_target.pkl')

# Load the test data
test_data = pd.read_csv(test_file)
test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
test_features = test_data.drop(columns=['PM25_Concentration'])
test_target = test_data['PM25_Concentration']

# Normalize test features
test_features_normalized = scaler_features.transform(test_features)
test_target_normalized = scaler_target.transform(test_target)

# Create sequences for testing
x_test, y_test = multivariate_data(test_features_normalized, test_target_normalized,
                                   start_index=0, end_index=None,
                                   history_size=step, target_size=1)

# Make predictions
predictions = trained_model.predict(x_test)

# Reshape y_test if needed
y_test = y_test.reshape(predictions.shape)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Save metrics to a file
with open(f'{city}_step_{step}_metrics.txt', 'w') as file:
    file.write(f"MSE without noise: {mse}\n")
    file.write(f"MAE without noise: {mae}\n")

# Display results in the terminal
print(f"Metrics for {city} with step size {step} using test file '{test_file}':")
print(f"MSE without noise: {mse}")
print(f"MAE without noise: {mae}")
