# gru_stock_price_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 1️⃣ Load Dataset
data = pd.read_csv("data/stock_data.csv")

# Filter data for one specific stock (e.g., Apple)
data = data[data['Ticker'] == 'AAPL']

# Drop rows with missing values
data = data.dropna()

# Select features and target
# Using Open, High, Low as input features
X = data[['Open Price', 'High Price', 'Low Price']].values
y = data['Close Price'].values

print("\nColumns in dataset:")
print(data.columns)

# 2️⃣ Normalize the features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1))

# 3️⃣ Create sequences for GRU (time series)
sequence_length = 60  # using past 60 days to predict next
X_seq, y_seq = [], []

for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# 4️⃣ Split into train and test sets
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# 5️⃣ Build GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 6️⃣ Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 7️⃣ Make predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# 8️⃣ Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred)
accuracy = r2 * 100  # approximate accuracy

print("\n📈 Model Evaluation Results:")
print(f"RMSE  : {rmse:.3f}")
print(f"MAE   : {mae:.3f}")
print(f"MAPE  : {mape:.2f}%")
print(f"R² Score: {r2:.3f}")
print(f"Approx. Accuracy: {accuracy:.2f}%")

# Save results to DataFrame
results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE', 'R² Score', 'Accuracy (%)'],
    'Value': [rmse, mae, mape, r2, accuracy]
})
print("\n📊 Final Model Results Table:")
print(results)

# 9️⃣ Visualization - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:100], label='Actual Price', color='blue', marker='o', linestyle='dashed')
plt.plot(y_pred[:100], label='Predicted Price', color='red', marker='x')
plt.title('Stock Market Price Prediction using GRU')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 10️⃣ Visualization - Residual Errors
residuals = y_test_actual - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residual Error Plot (Actual - Predicted)")
plt.xlabel("Sample Index")
plt.ylabel("Residual Error")
plt.grid(True)
plt.show()

# 11️⃣ Predict Next Day Price (based on last known sequence)
last_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
predicted_next_scaled = model.predict(last_sequence)
predicted_next_price = scaler_y.inverse_transform(predicted_next_scaled)[0][0]
print(f"\n💰 Predicted Next Day Close Price: {predicted_next_price:.2f}")

# Optional: Save results to CSV
results.to_csv("gru_results_summary.csv", index=False)
print("\n📁 Results saved to gru_results_summary.csv successfully!")
