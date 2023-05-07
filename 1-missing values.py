import pandas as pd

# Load the dataset
data = pd.read_csv("train.csv")

data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].median())
data.to_csv('train_missing.csv', index=False)

# Load the dataset
data = pd.read_csv("test.csv")

data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].median())
data.to_csv('test_missing.csv', index=False)

