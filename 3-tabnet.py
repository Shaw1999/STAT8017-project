import pandas as pd
import time
from sklearn.metrics import classification_report, accuracy_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess.csv')
test_data = pd.read_csv('test_preprocess.csv')

# Split the dataset into training and testing sets
X_train = train_data.drop('satisfaction', axis=1).values
X_test = test_data.drop('satisfaction', axis=1).values
y_train = train_data['satisfaction'].values
y_test = test_data['satisfaction'].values

# Set fixed hyperparameters
n_d = 8
n_a = 8
n_steps = 3
gamma = 1.3
n_independent = 2
n_shared = 2
lr = 0.02

# Train the model
clf = TabNetClassifier(
    n_d=n_d, n_a=n_a,
    n_steps=n_steps,
    gamma=gamma,
    n_independent=n_independent,
    n_shared=n_shared,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=1,
    optimizer_params=dict(lr=lr),
    device_name='cuda' if torch.cuda.is_available() else 'cpu'
)

start_time = time.time()
clf.fit(X_train, y_train, max_epochs=100, patience=10, batch_size=256)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds\n")

# Predict and evaluate
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save results to file
with open("3-tabnet-result.txt", "w") as file:
    file.write("Fixed hyperparameters:\n")
    file.write(f"n_d: {n_d}, n_a: {n_a}, n_steps: {n_steps}, gamma: {gamma}, n_independent: {n_independent}, n_shared: {n_shared}, lr: {lr}\n")
    file.write("Classification report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Training time: {end_time - start_time:.2f} seconds\n")

# Save the best model
clf.save_model('tabnet_best_model')

# Feature importance
feature_importances = clf.feature_importances_
feature_names = train_data.drop('satisfaction', axis=1).columns

sorted_indices = np.argsort(np.abs(feature_importances))[::-1]

with open('3-tabnet-result.txt', 'a') as f:
    f.write("\nTop 5 features impacting passenger satisfaction:\n")
    for index in sorted_indices[:5]:
        f.write(f"{feature_names[index]}: {feature_importances[index]}\n")

# Load the best model
clf_loaded = TabNetClassifier()
clf_loaded.load_model('tabnet_best_model')

