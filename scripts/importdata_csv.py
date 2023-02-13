# Import data part of project 1 - based on code from the following files: exercise 1.5.1
import numpy as np
import pandas as pd

# Load the train csv data using the Pandas library
filename = '../data/Dry_Bean_Dataset.CSV'
df = pd.read_csv(filename,sep=";")

# Replace , with .

# print(df.get('Area'))
# df.Area = df.Area.str.replace(",", '.')
# df.AspectRation = df.AspectRation.str.replace(",", '.')
# df.Compactness = df.Compactness.str.replace(",", '.')
# df.ConvexArea = df.ConvexArea.str.replace(",", '.')


# Convert pandas dataframe to numpy array
raw_data = df.values

# Create data matrix
cols = range(0, 16)
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# Extract labels from csvx
classLabels = raw_data[:, -1]  # -1 takes the last column

# Determine unique
# 0, 1, 2, 3 denotes price ranges
classNames = np.unique(classLabels)

# Create dictionary
classDict = dict(zip(classNames, range(len(classNames))))

# Add to dictionary
y = np.array([classDict[cl] for cl in classLabels])

# Get data objects and attributes from X dimensions
N, M = X.shape

# Number of classes
C = len(classNames)
