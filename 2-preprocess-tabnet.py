import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv('train_missing.csv')
test_data = pd.read_csv('test_missing.csv')

# Preprocess categorical variables using LabelEncoder
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

# Save preprocessed data
train_data.to_csv('train_preprocess_tabnet.csv', index=False)
test_data.to_csv('test_preprocess_tabnet.csv', index=False)
