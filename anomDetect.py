
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load industrial sensor data (Example: CSV format)
df = pd.read_csv("sensor_data.csv")

# Assume time-series data with multiple sensors
# Columns: ['timestamp', 'sensor1', 'sensor2', ..., 'sensorN']
df.set_index('timestamp', inplace=True)
df = df.astype(float)

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Create sequences for LSTM
SEQ_LEN = 50  # Adjust sequence length based on system dynamics

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

X_train = create_sequences(df_scaled, SEQ_LEN)

# Split into training and test sets
split_idx = int(0.8 * len(X_train))
X_train, X_test = X_train[:split_idx], X_train[split_idx:]

# Define LSTM Autoencoder
latent_dim = 16  # Bottleneck dimension

inputs = Input(shape=(SEQ_LEN, df.shape[1]))
encoded = LSTM(latent_dim, activation="relu", return_sequences=False)(inputs)
decoded = RepeatVector(SEQ_LEN)(encoded)
decoded = LSTM(df.shape[1], activation="relu", return_sequences=True)(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train Autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Predict on test data
X_test_pred = autoencoder.predict(X_test)

# Calculate reconstruction error
mse = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))

# Set anomaly detection threshold (e.g., mean + 2*std)
threshold = np.mean(mse) + 2 * np.std(mse)

# Detect anomalies
anomalies = mse > threshold

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.scatter(np.where(anomalies), mse[anomalies], color='red', label="Anomalies")
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.title("Anomaly Detection for Predictive Maintenance")
plt.show()

