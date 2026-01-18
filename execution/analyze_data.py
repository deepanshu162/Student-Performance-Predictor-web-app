import pandas as pd
import os

def analyze_student_performance():
    file_path = 'StudentPerformance.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
        print("Data Loaded Successfully.")
        print("-" * 20)
        print("First 5 rows:")
        print(df.head())
        print("-" * 20)
        print("\nBasic Statistics:")
        print(df.describe())
        print("-" * 20)
        print("\nMissing Values:")
        print(df.isnull().sum())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_student_performance()
