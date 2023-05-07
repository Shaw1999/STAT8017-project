import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-2]
X_test = test_data.iloc[:, :-2]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Set up logistic regression model and perform hyperparameter tuning with GridSearchCV
logistic_regression = LogisticRegression(max_iter=10000, random_state=42, C=0.1)



start_time = time.time()
logistic_regression.fit(X_train, y_train)
end_time = time.time()

print(f"Training time: {(end_time - start_time):.2f} seconds")

