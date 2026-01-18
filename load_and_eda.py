import pandas as pd
import os

def load_and_eda():
    file_path = 'StudentPerformance.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
        print("Data Loaded Successfully.")
        print(f"Shape: {df.shape}")
        print("-" * 20)
        print("Data Types:")
        print(df.dtypes)
        print("-" * 20)
        print("Missing Values:")
        print(df.isnull().sum())
        print("-" * 20)
        print("Summary Statistics:")
        print(df.describe(include='all'))
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    load_and_eda()
