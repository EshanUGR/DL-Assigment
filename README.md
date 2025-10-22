📈 Stock Price Forecasting using a 1D Convolutional Neural Network (CNN)

This repository focuses on applying a 1-Dimensional Convolutional Neural Network (1D CNN) to forecast the closing price of the BAJAJ-AUTO stock. 
The project demonstrates advanced time series preparation techniques and uses a standard CNN architecture tailored for sequential data.


🌟 Project Highlights

Model: 1D Convolutional Neural Network.

Target Data: BAJAJ-AUTO.csv historical stock prices.

Input Features: The model utilizes five key features for prediction: Open, High, Low, Close, and Volume.

Time Series Preparation: Data is structured into sequential windows of 60 timesteps.

Tuned Architecture: The CNN is trained with tuned hyperparameters (e.g., 50 epochs, learning rate 0.0005, and Dropout 0.1) to achieve an optimized fit.

Evaluation: Performance is measured using Mean Squared Error (MSE) and R2 Score.

Practical Application: Includes logic to predict the closing price for the next single trading day.
