import pandas as pd

try:
    df = pd.read_csv("./data/labels.csv")
    print(df.head())  # Print first few rows to confirm it's loading correctly
except FileNotFoundError:
    print("File not found. Please check the path.")
