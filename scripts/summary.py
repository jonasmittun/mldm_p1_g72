from matplotlib import pyplot as plt

from importdata_csv import *
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel,
                               xticks, yticks,legend,show)


# attributes_to_be_analysed = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
# attributeName_to_index = {}
# for i, att in enumerate(attributeNames):
#     if att in attributes_to_be_analysed:
#         attributeName_to_index[att] = i
#
# print(attributeName_to_index)
#
# # Mean
# summaries = {}
# for att in attributes_to_be_analysed:
#     mean = np.mean(X[:,attributeName_to_index[att]])
#     var = np.var(X[:,attributeName_to_index[att]])
#     sd = np.std(X[:,attributeName_to_index[att]])
#     summaries[att] = (mean, var, sd)

## Classification problem
# The current variables X and y represent a classification problem, in
# which a machine learning model will use the sepal and petal dimesions
# (stored in the matrix X) to predict the class (species of Iris, stored in
# the variable y). A relevant figure for this classification problem could
# for instance be one that shows how the classes are distributed based on
# two attributes in matrix X:

# Sus constellation
# x_att = 'mobile_wt'
# y_att = 'm_dep'

# plot_pairs = [('ram','pc'),('mobile_wt','battery_power'),('n_cores','ram')]



figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

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

#plt.figure("box_outliers")
#plt.boxplot(X[:,attributeName_to_index['battery_power']])
#plt.show()



#  Are there issues with outliers in the data,
#  do the attributes appear to be normal distributed,
#  are variables correlated,
#  does the primary machine learning modeling aim appear to be feasible based on your visualizations.

