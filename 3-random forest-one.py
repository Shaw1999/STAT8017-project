import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-2]
X_test = test_data.iloc[:, :-2]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Set up Random Forest classifier and perform hyperparameter tuning with GridSearchCV
rf_classifier = RandomForestClassifier(random_state=42,max_depth=20, min_samples_leaf=1, n_estimators=200)

start_time = time.time()
rf_classifier.fit(X_train,y_train)
end_time = time.time()


print(f"Training time: {end_time - start_time:.2f} seconds\n")

