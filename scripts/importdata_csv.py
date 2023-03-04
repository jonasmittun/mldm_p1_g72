# Import data part of project 1 - based on code from the following files: exercise 1.5.1
import numpy as np
import pandas as pd

# Load the train csv data using the Pandas library
filename = '../data/glass.data'
df = pd.read_csv(filename, sep=",")

num_of_att = 10

# Convert pandas dataframe to numpy array
raw_data = df.values

# Create data matrix
cols = range(1, num_of_att)
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Extract labels from csvx
classLabels = raw_data[:, -1]  # -1 takes the last column
classLabels[classLabels == 3] = 1
classLabels[classLabels > 4] = 3
newClassNames = ["float window", "non-float window","non-window"]
newClassLabels = [0 for _ in classLabels]
for i in range(len(classLabels)):
    newClassLabels[i] = newClassNames[int(classLabels[i])-1]

# print(classLabels, 'Num Observations', len(classLabels))

# Determine unique
classNames = np.unique(newClassLabels)

# Create dictionary
classDict = dict(zip(classNames, range(len(classNames))))


# Add to dictionary
y = np.array([classDict[cl] for cl in newClassLabels])

# Get data objects and attributes from X dimensions
N, M = X.shape

# Number of classes
C = len(classNames)

## Removes the potassium "outliers"
# SX = X - np.ones((N, 1)) * X.mean(0)
# SX = SX * (1 / np.std(SX, 0))

# outlier_mask = (SX[:, 5] > 8)
# valid_mask = np.logical_not(outlier_mask)
#
# # Finally we will remove these from the data set
# X = X[valid_mask, :]
# y = y[valid_mask]
# N = len(y)


# Defining class colors
class_colors = ['#8dddd0', 'darkgreen', '#ca472f']
