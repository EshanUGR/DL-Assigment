# main.py
from preprocessing import load_and_preprocess_data
from exploration import explore_data
from training import build_gru_model, train_model
from evaluation import evaluate_model, visualize_results, predict_next_day

# Parameters
sequence_length = 60
file_path = r"C:\Desktop\DL_ASSIGMENT\data\ADANIPORTS.csv"

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test, scaler_X, scaler_y, data = load_and_preprocess_data(
    file_path, sequence_length=sequence_length
)
print("X_train shape:", X_train.shape)

# Step 2: Explore data
explore_data(data)

# Step 3: Build and train GRU model
model = build_gru_model((X_train.shape[1], X_train.shape[2]))
model, history = train_model(model, X_train, y_train, X_test, y_test)

# Step 4: Evaluate and visualize
y_pred, y_test_actual, mse, mae, r2 = evaluate_model(model, X_test, y_test, scaler_y)
visualize_results(y_test_actual, y_pred)

# Step 5: Predict next day price
last_sequence = scaler_X.transform(data[['Open', 'High', 'Low']].values)[-sequence_length:]
predict_next_day(model, last_sequence, scaler_y)
