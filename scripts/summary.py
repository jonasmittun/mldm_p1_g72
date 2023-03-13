## Author: Group 72

from matplotlib import pyplot as plt

from importdata_csv import *
from matplotlib.pyplot import (figure, subplot, xlabel, ylabel,
                               xticks, yticks, legend, show, hist, title,
                               subplots_adjust, scatter, savefig)

# Compute basic summary statistics
summaries = {}
for i,att in enumerate(attributeNames):
    col = X[:, i]
    mean = np.mean(col)
    var = np.var(col)
    sd = np.std(col)
    summaries[att] = (mean, var, sd)

# Histograms
figure(figsize=(12, 10))
for i,att in enumerate(attributeNames):
    subplot(3, 3, i+1)
    hist(X[:, i], bins=30)
    title(att)
    subplots_adjust(hspace=0.5, wspace=0.5)
show()


# Histogram over classes
figure(figsize=(12, 10))
for i,att in enumerate(attributeNames):
    subplot(3, 3, i+1)
    hist(X[:, i], bins=30)
    title(att)
    subplots_adjust(hspace=0.5, wspace=0.5)
show()


# Normalize data
SX = X - np.ones((N, 1)) * X.mean(0)
SX = SX * (1 / np.std(SX, 0))

# Attribute scatter plot (box style)
figure(figsize=(9, 6))
for i, att in enumerate(attributeNames):
    for c in range(C):
        class_mask = (y == c)
        noise = np.random.uniform(low=-0.25, high=0.25, size=len(SX[class_mask, 0]))
        scatter(i+1+noise, SX[class_mask, i], marker='.', s=8, c=class_colors[c],alpha=0.50)
xticks(range(1, len(attributeNames)+1), attributeNames)
title('Attribute scatter plot (Box style)')
ylabel('Standardized attribute values')
legend(classNames)
# savefig("../plots/spbox-050alpha.svg",bbox_inches='tight')
show()

# Matrix scatter plot of attributes
r_mask = np.ones(N, dtype=bool)
figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = ((y == c) & r_mask)
            scatter(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), marker='.', s=8, c=class_colors[c], alpha=0.50)
            if m1 == M-1:
                xticks(fontsize=7)
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2 == 0:
                yticks(fontsize=7)
                ylabel(attributeNames[m1])
            else:
                yticks([])
subplot(M, M, 1).legend(classNames, bbox_to_anchor=(2, 2.3), loc='upper right')
plt.suptitle('Attribute scatter plot matrix', fontsize=14)
# savefig("../plots/spm-050alpha.svg", bbox_inches='tight')
show()
