import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-1]
X_test = test_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Set up decision tree classifier and perform hyperparameter tuning with GridSearchCV
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_leaf=20)

start_time = time.time()
dt_classifier.fit(X_train, y_train)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds\n")

