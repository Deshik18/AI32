import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import joblib
import os

# Create a new folder to store results
result_folder = 'results'
os.makedirs(result_folder, exist_ok=True)

# Load and preprocess your training data for each city (replace 'your_data.csv' with actual file names)
cities = ['B', 'G', 'S', 'T']
step_sizes = [1, 7, 14, 30, 60]


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, 1)
        data.append(dataset[indices])

        if target_size == 1:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


for city in cities:
    # Normalize data
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    for step in step_sizes:
        # Load data
        df = pd.read_csv(f'C:/Users/deshik/ml/Sample_DataSets/{city}_train.csv')
        target_column = 'PM25_Concentration'

        # Drop unnecessary columns
        df = df.drop(columns=['Unnamed: 0'], axis=1)

        # Extract features and target
        features = df.drop(columns=[target_column])
        target = df[target_column]

        features_normalized = scaler_features.fit_transform(features)
        target_normalized = scaler_target.fit_transform(target.values.reshape(-1, 1))

        STEP = step

        # Create sequences for multi-step forecasting
        x_train, y_train = multivariate_data(features_normalized, target_normalized,
                                             start_index=0, end_index=None,
                                             history_size=step, target_size=1)

        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

        model = Sequential([
            Bidirectional(GRU(128, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.5),
            Bidirectional(GRU(64, return_sequences=False)),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(16, activation='relu'),  # Additional hidden layer
            Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Set up model checkpoints and early stopping
        checkpoint_path = f'C:/Users/deshik/ml/saved_models/{city}_model_{STEP}.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val),
                            callbacks=[checkpoint, early_stopping])

        trained_model = tf.keras.models.load_model(f'C:/Users/deshik/ml/saved_models/{city}_model_{step}.h5')

        # Load the test data
        test_data = pd.read_csv(f'C:/Users/deshik/ml/Sample_DataSets/{city}_test.csv')
        test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
        test_features = test_data.drop(columns=['PM25_Concentration'])
        test_target = test_data['PM25_Concentration']

        # Normalize test features
        test_features_normalized = scaler_features.transform(test_features)
        test_target_normalized = scaler_target.transform(test_target.values.reshape(-1, 1))

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

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.plot(y_test[:3].flatten(), label='True Values')
        plt.plot(predictions[:3].flatten(), label='Predicted Values')
        plt.title('Sample Predictions')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        sample_graphs_path = os.path.join(result_folder, f'{city}_step_{STEP}_sample_graphs.png')
        plt.savefig(sample_graphs_path)

        # Save metrics in the results folder
        metrics_path = os.path.join(result_folder, f'{city}_step_{STEP}_metrics.txt')
        with open(metrics_path, 'w') as file:
            file.write(f"MSE without noise: {mse}\n")
            file.write(f"MAE without noise: {mae}\n")

    joblib.dump(scaler_target, f'C:/Users/deshik/ml/saved_models/{city}_scaler_target.pkl')
    joblib.dump(scaler_features, f'C:/Users/deshik/ml/saved_models/{city}_scaler_features.pkl')