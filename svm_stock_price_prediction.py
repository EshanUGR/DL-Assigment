# svm_stock_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1️⃣ Load Dataset
data = pd.read_csv("data/stock_data.csv")

# Filter data for one specific stock (e.g., Apple)
data = data[data['Ticker'] == 'AAPL']

# Drop rows with missing values
data = data.dropna()

# Select features and target
X = data[['Open Price', 'High Price', 'Low Price']]
y = data['Close Price']


print("\nColumns in dataset:")
print(data.columns)

# 3️⃣ Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5️⃣ Train SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# 6️⃣ Make predictions
y_pred = svm_model.predict(X_test)

# 7️⃣ Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100  # Approximate accuracy

print("\n📈 Model Evaluation Results:")
print(f"RMSE  (Root Mean Squared Error): {rmse:.3f}")
print(f"MAE   (Mean Absolute Error): {mae:.3f}")
print(f"MAPE  (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"R² Score: {r2:.3f}")
print(f"✅ Model Accuracy (approx.): {accuracy:.2f}%")

# Save results to DataFrame (for report use)
results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE', 'R² Score', 'Accuracy (%)'],
    'Value': [rmse, mae, mape, r2, accuracy]
})
print("\n📊 Final Model Results Table:")
print(results)

# 8️⃣ Visualization - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual Price', color='blue', marker='o', linestyle='dashed')
plt.plot(y_pred[:100], label='Predicted Price', color='red', marker='x')
plt.title('Stock Market Price Prediction using SVM')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 9️⃣ Visualization - Residual Errors
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residual Error Plot (Actual - Predicted)")
plt.xlabel("Sample Index")
plt.ylabel("Residual Error")
plt.grid(True)
plt.show()

# 🔟 Predict Next Day Price (based on the last known values)
last_row = X_scaled[-1].reshape(1, -1)
predicted_next_price = svm_model.predict(last_row)[0]
print(f"\n💰 Predicted Next Day Close Price: {predicted_next_price:.2f}")

# ✅ Optional: Save model results to CSV (for report inclusion)
results.to_csv("svm_results_summary.csv", index=False)
print("\n📁 Results saved to svm_results_summary.csv successfully!")
