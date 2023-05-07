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
logistic_regression = LogisticRegression(max_iter=10000, random_state=42)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()


# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)

# Determine feature importance
coefs = grid_search.best_estimator_.coef_[0]
sorted_indices = np.argsort(np.abs(coefs))[::-1]
feature_names = X_train.columns

with open("3-lr-result.txt", "w") as file:
    file.write(f"Best hyperparameter: {grid_search.best_params_}")
    file.write(f"Training time: {(end_time - start_time):.2f} seconds")
    file.write("Accuracy score on the test set: \n{:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    file.write("Classification report:")
    file.write(classification_report(y_test, y_pred))
    file.write("Top 5 features impacting passenger satisfaction:")
    for index in sorted_indices[:5]:
        file.write(f"{feature_names[index]}: {coefs[index]}")
