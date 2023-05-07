import pandas as pd
import time

import xgboost as xgb

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]


# Set up XGBClassifier and perform hyperparameter tuning with GridSearchCV
xgb_classifier = xgb.XGBClassifier(random_state=42, learning_rate=0.2, max_depth=20, n_estimators=200)


start_time = time.time()
xgb_classifier.fit(X_train, y_train)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds\n")

