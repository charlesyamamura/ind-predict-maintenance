#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 08:43:54 2025

@author: charlesyamamura
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
import queue
from sklearn.preprocessing import MinMaxScaler

# Load trained LSTM Autoencoder
model = load_model("lstm_autoencoder.h5")

# Load and normalize initial sensor data for reference
df = pd.read_csv("sensor_data.csv")
df.set_index("timestamp", inplace=True)
df = df.astype(float)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define real-time streaming parameters
SEQ_LEN = 50
BUFFER = []  # Rolling buffer to store last SEQ_LEN sensor readings
DATA_QUEUE = queue.Queue()  # Queue to simulate real-time data arrival

# Define anomaly detection threshold based on training set
X_train = np.array([df_scaled[i : i + SEQ_LEN] for i in range(len(df_scaled) - SEQ_LEN)])
train_preds = model.predict(X_train)
mse = np.mean(np.abs(train_preds - X_train), axis=(1, 2))
THRESHOLD = np.mean(mse) + 2 * np.std(mse)

# Function to process real-time data
def process_real_time_data():
    global BUFFER
    
    while True:
        if not DATA_QUEUE.empty():
            new_data = DATA_QUEUE.get()
            BUFFER.append(new_data)

            # Maintain buffer size
            if len(BUFFER) > SEQ_LEN:
                BUFFER.pop(0)

            if len(BUFFER) == SEQ_LEN:  # Only analyze when buffer is full
                input_data = np.array(BUFFER).reshape(1, SEQ_LEN, -1)
                pred = model.predict(input_data)
                error = np.mean(np.abs(pred - input_data))

                if error > THRESHOLD:
                    print(f"🚨 Anomaly detected! Reconstruction error: {error:.4f} (Threshold: {THRESHOLD:.4f})")

# Start real-time processing thread
processing_thread = threading.Thread(target=process_real_time_data, daemon=True)
processing_thread.start()

# Simulate real-time sensor data streaming
sensor_columns = df.columns

for _ in range(500):  # Simulate 500 incoming sensor readings
    simulated_sensor_reading = np.random.normal(size=(len(sensor_columns)))  # Simulated random sensor data
    scaled_reading = scaler.transform(simulated_sensor_reading.reshape(1, -1)).flatten()
    DATA_QUEUE.put(scaled_reading)
    
    time.sleep(0.1)  # Simulate real-time delay (adjust as needed)
