# Real-Time Anomaly Detection with LSTM Autoencoder

This repository implements a real-time anomaly detection system using a pre-trained LSTM autoencoder. The model monitors multivariate time series sensor data and flags anomalies based on reconstruction error.

## ⚙️ System Overview

- **Model Type**: LSTM Autoencoder
- **Use Case**: Anomaly detection in streaming sensor data
- **Data Source**: `sensor_data.csv`
- **Frameworks**: TensorFlow, NumPy, Pandas, Scikit-learn
- **Streaming Simulation**: Python threading and queue-based emulation

## 🧠 Model Details

- The autoencoder is trained to reconstruct normal behavior patterns from time series sequences.
- Anomalies are detected when the reconstruction error exceeds a statistically defined threshold.
- The threshold is calculated using training set MSE:
  \[
  \text{THRESHOLD} = \mu + 2\sigma
  \]

## 📁 Files

- `main.py` – Core script for real-time simulation and anomaly detection.
- `lstm_autoencoder.h5` – Pre-trained LSTM autoencoder model.
- `sensor_data.csv` – Multivariate time series sensor dataset with a `timestamp` column.

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn
