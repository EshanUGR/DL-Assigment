# evaluation.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, scaler_y):
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_actual, predictions)
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

    print("\n📈 Model Evaluation Results:")
    print(f"MSE  : {mse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    return predictions, y_test_actual, mse, mae, r2

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Price', color='blue', marker='o', linestyle='dashed')
    plt.plot(predictions, label='Predicted Price', color='red', marker='x')
    plt.title('Stock Price Prediction using GRU')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_next_day(model, last_sequence, scaler_y):
    last_sequence = np.expand_dims(last_sequence, axis=0)  # (1, timesteps, features)
    next_pred = model.predict(last_sequence)
    next_pred = scaler_y.inverse_transform(next_pred)
    print(f"\n💰 Predicted Next Day Close Price: {next_pred[0][0]:.2f}")
    return next_pred[0][0]
