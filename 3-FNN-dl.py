import pandas as pd
import time
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV

# Load preprocessed training and testing data
train_data = pd.read_csv('train_preprocess_dl.csv')
test_data = pd.read_csv('test_preprocess_dl.csv')

# Split the dataset into training and testing sets
X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.astype(np.float32)
y_test = test_data.iloc[:, -1].values.astype(np.float32)

# Create FNN model with PyTorch
class FNN(nn.Module):
    def __init__(self, input_dim, layers, activation):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        for i, nodes in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, nodes))
            else:
                self.layers.append(nn.Linear(layers[i-1], nodes))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        x = x.to(self.layers[0].weight.dtype)  # Convert input tensor to the same dtype as weights
        for layer in self.layers:
            x = layer(x)
        return x

# Custom FNN Classifier
class FNNClassifier(NeuralNetBinaryClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_X(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return X_tensor

    def preprocess_y(self, y):
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return y_tensor

    def fit(self, X, y, **fit_params):
        X = self.preprocess_X(X)
        y = self.preprocess_y(y)
        return super().fit(X, y, **fit_params)


# Hyperparameters for tuning
param_grid = {
    'module__layers': [(45, 30, 15)],
    'module__activation': [nn.ReLU()],
    'batch_size': [128],
    'max_epochs': [30],
    'lr': [0.001],
    'optimizer__weight_decay': [0.1]
}

# Perform hyperparameter tuning with GridSearchCV
torch.manual_seed(42)
fnn = FNNClassifier(
    FNN,
    module__input_dim=X_train.shape[1],
    criterion=nn.BCELoss,
    optimizer=optim.Adam,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

grid_search = GridSearchCV(fnn, param_grid, cv=3, n_jobs=-1, scoring='accuracy', error_score='raise')
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save results to file
with open("3-FNN-result_dl.txt", "w") as file:
    file.write(f"Best hyperparameter: {grid_search.best_params_}\n")
    file.write(f"Training time: {end_time - start_time:.2f} seconds\n")
    file.write("Classification report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write("Feature importance is not directly available for FNN.\n")

# Save the best model
torch.save(grid_search.best_estimator_.module_.state_dict(), 'best_fnn_model_dl.pt')

# Load the best model for evaluation
best_fnn_model = FNN(input_dim=X_train.shape[1], layers=grid_search.best_params_['module__layers'], activation=grid_search.best_params_['module__activation'])
best_fnn_model.load_state_dict(torch.load('best_fnn_model_dl.pt'))

# Convert the test data into PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Set the model to evaluation mode
best_fnn_model.eval()

# Forward the test data through the model to get predictions
y_pred_proba = best_fnn_model(X_test_tensor)

# Convert the predictions to binary values based on a threshold
threshold = 0.5
y_pred = (y_pred_proba > threshold).float().detach().numpy()

