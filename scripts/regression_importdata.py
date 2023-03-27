## Author: Group 72

# Import data part of project 1 - based on code from the following files: exercise 1.5.1
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, hist, xlabel, ylabel, title, show

# Load the train csv data using the Pandas library
filename = '../data/glass.data'
df = pd.read_csv(filename, sep=",")

num_of_att = 9

# Convert pandas dataframe to numpy array
raw_data = df.values


# Create data matrix
cols = range(2, 10)
X = raw_data[:, cols]

# Add to dictionary
y = raw_data[:, 1]

# Attribute names
attributeNames = np.asarray(df.columns[cols])

# Get data objects and attributes from X dimensions
N, M = X.shape

# Standardize data
X = X - np.ones((N, 1)) * X.mean(0)
X = X * (1 / np.std(X, 0))

y = y - y.mean()
y = y * (1 / np.std(y))

