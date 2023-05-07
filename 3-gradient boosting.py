import pandas as pd
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Set up XGBClassifier and perform hyperparameter tuning with GridSearchCV
xgb_classifier = xgb.XGBClassifier(random_state=42, tree_method='gpu_hist', gpu_id=0)

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [1, 5, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2, 0.3]
}

grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, n_jobs=-1)
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Determine feature importance
importances = grid_search.best_estimator_.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feature_names = X_train.columns

# Save results to file
with open("3-gradient boosting-result.txt", "w") as file:
    file.write(f"Best hyperparameter: {grid_search.best_params_}\n")
    file.write(f"Training time: {end_time - start_time:.2f} seconds\n")
    file.write("Classification report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write("Top 5 features impacting passenger satisfaction:\n")
    for index in sorted_indices[:5]:
        file.write(f"{feature_names[index]}: {importances[index]}\n")
