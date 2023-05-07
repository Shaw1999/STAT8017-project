import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

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

# Perform PCA on continuous features and find the best number of components
pca = PCA()
pca.fit(scaled_continuous_data)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Get the PCA coefficients (eigenvectors)
pca_coefficients = pca.components_

best_n_components = np.argmax(explained_variance_ratio > 0.95) + 1
pca = PCA(n_components=best_n_components)
principal_components = pca.fit_transform(scaled_continuous_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(best_n_components)])

# Encode categorical variables
encoded_categorical_data = pd.get_dummies(data[categorical_columns])

# Combine PCA, encoded categorical data and Satisfaction
satisfaction = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
preprocessed_data = pd.concat([principal_df, encoded_categorical_data, satisfaction], axis=1)

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv('train_preprocess.csv', index=False)

# Load the dataset
data = pd.read_csv('test_missing.csv')

# Scale continuous features using MinMaxScaler
scaled_continuous_data = pd.DataFrame(scaler.fit_transform(data[continuous_columns]), columns=continuous_columns)

# Perform PCA on continuous features and find the best number of components
principal_components = pca.fit_transform(scaled_continuous_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(best_n_components)])

# Encode categorical variables
encoded_categorical_data = pd.get_dummies(data[categorical_columns])

# Combine PCA, encoded categorical data and Satisfaction
satisfaction = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
preprocessed_data = pd.concat([principal_df, encoded_categorical_data, satisfaction], axis=1)

# Save the preprocessed data to a CSV file
preprocessed_data.to_csv('test_preprocess.csv', index=False)
