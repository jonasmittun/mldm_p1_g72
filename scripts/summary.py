import numpy as np
from matplotlib import pyplot as plt

from importdata_csv import *
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, scatter,
                               xticks, yticks, legend, show, hist, title, subplots_adjust, savefig)

# Dictionary of indices of attributes #
attributes_to_be_analysed = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
attributeName_to_index = {}
for i, att in enumerate(attributeNames):
    if att in attributes_to_be_analysed:
        attributeName_to_index[att] = i

print(attributeName_to_index)

# Compute mean, variance and standard deviation #
summaries = {}
for att in attributes_to_be_analysed:
    minn = np.min(X[:, attributeName_to_index[att]])
    maxx = np.max(X[:, attributeName_to_index[att]])
    mean = np.mean(X[:, attributeName_to_index[att]])
    sd = np.std(X[:, attributeName_to_index[att]])
    summaries[att] = (att, minn, maxx, mean, sd)


def summaryToLatexTable(summary):
    # print(summary)
    for att in summary:
        stats = summary[att]
        print("{0} & {1} & {2} & {3} & {4} \\\\\\hline".format(stats[0],
                                                               round(stats[1], 4),
                                                               round(stats[2], 4),
                                                               round(stats[3], 4),
                                                               round(stats[4], 4)))


print(summaries)
summaryToLatexTable(summaries)

# Covariance and correlation coefficient matrices as recommended in the exercise 3 text
cov = np.cov(X.T)  # We transpose since the functions assume rows are variables, not columns like in our case
corr = np.corrcoef(X.T)

# ## Plot histograms ## Written by Tobias, sorry for bad code
figure(figsize=(12, 10))
for att in attributes_to_be_analysed:
    subplot(3, 3, attributeName_to_index[att] + 1)
    hist(X[:, attributeName_to_index[att]], bins=12)
    title(att)
    subplots_adjust(hspace=0.5, wspace=0.5)

# Plot all attributes against each other ##
figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = (y == c)
            scatter(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), s=5, marker='.')
            if m1 == M - 1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2 == 0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
legend(classNames)

savefig("scatter-plot-matrix.svg", format='svg')

show()

print('Ran Exercise 4.2.5')

# for x_att,y_att in plot_pairs:
#     x_l = X[:,attributeName_to_index[x_att]]
#     y_l = X[:,attributeName_to_index[y_att]]
#     plt.title(str(x_att)+" vs. "+str(y_att))
#     plt.scatter(x=x_l,
#                 y=y_l)
#     plt.xlabel(x_att)
#     plt.ylabel(y_att)
#     plt.show()

# X_c = X.copy();
# y_c = y.copy();
#
# plot_pairs = zip()
#
# attributeNames_c = attributeNames.copy();
# for x_att, y_att in plot_pairs:
#     i = attributeName_to_index[x_att]; j = attributeName_to_index[y_att];
#     color = ['r','g', 'b', 'y', 'c', 'm', 'k']
#     for c in range(len(classNames)):
#         idx = y_c == c
#         plt.scatter(x=X_c[idx, i],
#                     y=X_c[idx, j],
#                     c=color[c],
#                     s=50, alpha=0.5,
#                     label=classNames[c])
#     plt.legend()
#     plt.xlabel(attributeNames_c[i])
#     plt.ylabel(attributeNames_c[j])
#     plt.show()

# X_c = X.copy();
# y_c = y.copy();
# attributeNames_c = attributeNames.copy();
# i = 1; j = 2;
# color = ['r','g', 'b']
#
#
# for c in range(len(classNames)):
#     idx = y_c == c
#     plt.scatter(x=X_c[idx, i],
#                 y=X_c[idx, j],
#                 c=color[c],
#                 s=50, alpha=0.5,
#                 label=classNames[c])
# plt.legend()
# plt.xlabel(attributeNames_c[i])
# plt.ylabel(attributeNames_c[j])
# plt.show()

# plt.figure("box_outliers")
# plt.boxplot(X[:,attributeName_to_index['battery_power']])
# plt.show()


#  Are there issues with outliers in the data,
#  do the attributes appear to be normal distributed,
#  are variables correlated,
#  does the primary machine learning modeling aim appear to be feasible based on your visualizations.
