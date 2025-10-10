# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, ticker='AAPL', sequence_length=60):
    # Load dataset
    data = pd.read_csv(file_path)
   
    data = data.dropna()

    # Select features and target
    X = data[['Open', 'High', 'Low']].values
    y = data['Close'].values

    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Adjust sequence length if dataset is small
    sequence_length = min(sequence_length, len(X_scaled) - 1)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Train-test split
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, data
