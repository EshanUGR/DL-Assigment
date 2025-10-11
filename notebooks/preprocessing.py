# data_preprocessing.py

# Purpose: All functions related to cleaning and preparing your data.



# preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load CSV data from the given file path.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Drop unnecessary columns and remove missing values.
    
    Parameters:
        data (pd.DataFrame): Original data.
        
    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Drop columns
    data_new = data.drop(['Series','Trades','%Deliverble','Deliverable Volume','Turnover','Volume'], axis=1)
    
    # Drop missing values
    data_new = data_new.dropna()
    
    return data_new

def describe_data(data):
    """
    Print basic description and duplicates information.
    
    Parameters:
        data (pd.DataFrame): Data to describe.
    """
    print("Data Description:")
    print(data.describe())
    
    print("\nNumber of Duplicated Rows:", data.duplicated().sum())
    
    print("\nColumn Names:", data.columns.tolist())

def plot_prices(data):
    """
    Plot all relevant stock prices.
    
    Parameters:
        data (pd.DataFrame): Data to plot.
    """
    plt.figure(figsize=(12,6))
    
    for col in ['Prev Close','Open','High','Low','Last','Close','VWAP']:
        plt.plot(data[col], label=col)
    
    plt.legend()
    plt.xlabel("Date (Time)")
    plt.ylabel("Price ($)")
    plt.title("Stock Prices Over Time")
    plt.show()
