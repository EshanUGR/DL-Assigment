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





# # 5️⃣ Train SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)



# # 6️⃣ Make predictions
y_pred = svm_model.predict(X_test)

# # 7️⃣ Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# # 8️⃣ Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual Price', color='blue')
plt.plot(y_pred[:100], label='Predicted Price', color='red')
plt.title('Stock Market Price Prediction using SVM')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.show()

# # 9️⃣ Predict Next Day Price (example)
# # Using the last row in dataset as input
last_row = X_scaled[-1].reshape(1, -1)
predicted_next_price = svm_model.predict(last_row)[0]
print(f"\n💰 Predicted Next Day Close Price: {predicted_next_price:.2f}")
