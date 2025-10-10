# exploration.py
import matplotlib.pyplot as plt

def explore_data(data):
    print("\n📄 Columns in dataset:")
    print(data.columns)
    print("\n📈 Basic Statistics:")
    print(data.describe())

    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.title('Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()
