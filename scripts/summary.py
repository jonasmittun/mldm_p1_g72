from matplotlib import pyplot as plt

from importdata_csv import *
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel,
                               xticks, yticks, legend, show, hist, title,
                               subplots_adjust, scatter)

attributes_to_be_analysed = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
attributeName_to_index = {}
for i, att in enumerate(attributeNames):
    if att in attributes_to_be_analysed:
        attributeName_to_index[att] = i
# print(attributeName_to_index)

# Compute basic summary statistics
summaries = {}
for att in attributes_to_be_analysed:
    mean = np.mean(X[:, attributeName_to_index[att]])
    var = np.var(X[:, attributeName_to_index[att]])
    sd = np.std(X[:, attributeName_to_index[att]])
    summaries[att] = (mean, var, sd)

# Histograms
figure(figsize=(12, 10))
for att in attributes_to_be_analysed:
    subplot(3, 3, attributeName_to_index[att]+1)
    hist(X[:, attributeName_to_index[att]], bins=30)
    title(att)
    subplots_adjust(hspace=0.5, wspace=0.5)

# Normalize data
SX = X - np.ones((N, 1)) * X.mean(0)
SX = SX * (1 / np.std(SX, 0))

# Attribute scatter plot (box style)
figure()
for i, att in enumerate(attributes_to_be_analysed):
    for c in range(C):
        class_mask = (y == c)
        noise = np.random.uniform(low=-0.25, high=0.25, size=len(SX[class_mask, 0]))
        scatter(i+1+noise, SX[class_mask, attributeName_to_index[att]], marker='.', s=8, c=class_colors[c])
xticks(range(1, len(attributes_to_be_analysed)+1), attributes_to_be_analysed)
title('Attribute scatter plot')
ylabel('Standardized attribute values')
legend(classNames)

# Matrix scatter plot of attributes
r_mask = np.ones(N, dtype=bool)
# r_mask = np.random.uniform(size=N) < 0.1
figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = ((y == c) & r_mask)
            scatter(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), marker='.', s=8, c=class_colors[c])
            if m1 == M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2 == 0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
legend(classNames, bbox_to_anchor=(1.04, 0.5), loc="center left")

# Show all figures
show()
