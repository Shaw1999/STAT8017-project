import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("train.csv")

# Preprocessing: Convert satisfaction column to numerical values for plotting
data['satisfaction'] = data['satisfaction'].astype('category').cat.codes

# Select only numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = numerical_columns[2:]

# Determine the number of rows and columns for the plot grid
n_rows = len(numerical_columns) // 3
if len(numerical_columns) % 3 != 0:
    n_rows += 1

# Create a figure with specified size
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))

# Iterate through numerical columns and plot the distribution
for idx, col in enumerate(numerical_columns):
    row, col_idx = divmod(idx, 3)
    if col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        sns.histplot(data=data, x=col, kde=True, ax=axes[row, col_idx], binwidth=10)
        axes[row, col_idx].set_title(col)
    else:
        sns.histplot(data=data, x=col, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(col)
# Remove empty plots if there are any
if n_rows * 3 > len(numerical_columns):
    for i in range(len(numerical_columns), n_rows * 3):
        row, col_idx = divmod(i, 3)
        fig.delaxes(axes[row, col_idx])

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('1-distribution.png', dpi=300)
plt.show(block=True)
