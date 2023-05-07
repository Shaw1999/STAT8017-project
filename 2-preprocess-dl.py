import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('train_missing.csv')

# Identify continuous and categorical columns
continuous_columns = [
    'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
    'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Scale continuous features using MinMaxScaler
scaler = MinMaxScaler()
scaled_continuous_data = pd.DataFrame(scaler.fit_transform(data[continuous_columns]), columns=continuous_columns)

# Encode categorical variables
encoded_categorical_data = pd.get_dummies(data[categorical_columns])

# Combine PCA, encoded categorical data and Satisfaction
satisfaction = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
preprocessed_data = pd.concat([scaled_continuous_data, encoded_categorical_data, satisfaction], axis=1)

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv('train_preprocess_dl.csv', index=False)

# Load the dataset
data = pd.read_csv('test_missing.csv')

# Scale continuous features using MinMaxScaler
scaled_continuous_data = pd.DataFrame(scaler.fit_transform(data[continuous_columns]), columns=continuous_columns)

# Encode categorical variables
encoded_categorical_data = pd.get_dummies(data[categorical_columns])

# Combine PCA, encoded categorical data and Satisfaction
satisfaction = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
preprocessed_data = pd.concat([scaled_continuous_data, encoded_categorical_data, satisfaction], axis=1)

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv('test_preprocess_dl.csv', index=False)
