# svm_stock_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score





# 1️⃣ Load Dataset
data = pd.read_csv("data/stock_data.csv")



# Show first few rows
print("Dataset preview:")
print(data.head())




# 2️⃣ Select features and target

# Predict 'Close' price
data = data.dropna()
X = data[['Open Price', 'High Price', 'Low Price']]

y = data['Close Price']
print(data.columns)
# # 3️⃣ Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



#  4️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
