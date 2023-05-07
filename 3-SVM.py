import pandas as pd
import time
from sklearn.svm import SVC
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

# Set up SVM classifier and perform hyperparameter tuning with GridSearchCV
svm_classifier = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, n_jobs=-1)
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save results to file
with open("3-SVM-result.txt", "w") as file:
    file.write(f"Best hyperparameter: {grid_search.best_params_}\n")
    file.write(f"Training time: {end_time - start_time:.2f} seconds\n")
    file.write("Classification report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write("Feature importance is not directly available for SVM.\n")
