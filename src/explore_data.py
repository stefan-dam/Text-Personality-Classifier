import pandas as pd
import numpy as np

def explore_dataset():
    # Read the dataset
    df = pd.read_csv('data/data-final.csv')
    
    # Display basic information
    print("Dataset Information:")
    print("-" * 50)
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print("-" * 50)
    for col in df.columns:
        print(col)
    
    # Display first few rows
    print("\nFirst few rows of the dataset:")
    print("-" * 50)
    print(df.head())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print("-" * 50)
    print(df.describe())

if __name__ == '__main__':
    explore_dataset()
