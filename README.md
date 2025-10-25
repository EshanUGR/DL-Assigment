🧠 Stock Market Price Prediction – ANN Model

Author: Vimukthi P L D O
Student ID: IT22894656
Module: SE4050 – Deep Learning
Project Type: Supervised Learning (Regression)

📘 Project Overview

This repository contains the Artificial Neural Network (ANN) implementation for the group project “Stock Market Price Prediction using Deep Learning Models.”
The goal of this project is to predict the next-day closing price of a stock using real-world data from the NIFTY 50 Stock Market Dataset.

The ANN model serves as the baseline architecture for performance comparison with other models (1D-CNN, LSTM, and GRU) developed by other group members.

📊 Dataset

Dataset: NIFTY 50 Stock Market Data (Kaggle)

Description:
Daily trading records of 50 major Indian companies listed on the National Stock Exchange (NSE).

Main Attributes:

Date – Trading date

Open – Opening price

High – Highest price of the day

Low – Lowest price of the day

Close – Closing price

Volume – Total shares traded

Target Variable:

Next_Close – Next day’s closing price (created by shifting Close by one day)

⚙️ Methodology
1. Data Preprocessing

Imported data using pandas.read_csv()

Removed missing values

Selected relevant features: ['Open','High','Low','Close','Volume']

Created target column:

df['Next_Close'] = df['Close'].shift(-1)


Normalized all features and target using MinMaxScaler (0–1)

Split data chronologically: 80% training, 20% testing

2. Model Architecture
Layer	Type	Units	Activation
Input	Dense	128	ReLU
Hidden	Dense	64	ReLU
Dropout	—	0.2	—
Output	Dense	1	Linear

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Epochs: 100
Batch Size: 32
Callbacks: EarlyStopping, ReduceLROnPlateau

🧩 Implementation

Main file: ann_model.ipynb
Key libraries used:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

📈 Results
Metric	Value
RMSE	203.10
R² Score	0.7412

Interpretation:
The ANN model explained approximately 74% of the variance in the test data.
It performed well in identifying nonlinear patterns but was less effective in capturing long-term temporal dependencies compared to sequential models like LSTM and GRU.

💡 Result Visualization

Actual vs Predicted Closing Prices

Training vs Validation Loss Curve

(Plots included in the notebook.)

🔍 Comparison with Other Models
Model	Accuracy (%)	R² Score	RMSE
LSTM	96.79	0.9679	0.0180
GRU	95.63	0.954	72.90
1D-CNN	76.30	–	–
ANN	74.12	0.7412	203.10
🚀 Future Improvements

Integrate technical indicators (RSI, EMA, MACD)

Combine ANN with LSTM or GRU (hybrid architecture)

Hyperparameter tuning with Grid Search

Deploy as a web-based forecasting dashboard
