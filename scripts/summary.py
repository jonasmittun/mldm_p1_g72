from matplotlib import pyplot as plt

from importdata_xls import *

# Erroneous data
# sc_w - May contain 0's

# sc_h sc_w m_dep

# touch_screen

# Mean
# Battery power, clock speed, n_cores, primary camera mega pixels, ram,
# attributes_to_be_analysed = ['mobile_wt','battery_power','clock_speed','n_cores','int_memory','ram','pc', 'sc_h', 'sc_w', 'm_dep', 'touch_screen']
attributes_to_be_analysed = ['Area', 'Perimeter']
attributeName_to_index = {}
for i, att in enumerate(attributeNames):
    if att in attributes_to_be_analysed:
        attributeName_to_index[att] = i

print(attributeName_to_index)

# Mean
summaries = {}
for att in attributes_to_be_analysed:
    mean = np.mean(X[:,attributeName_to_index[att]])
    var = np.var(X[:,attributeName_to_index[att]])
    sd = np.std(X[:,attributeName_to_index[att]])
    summaries[att] = (mean, var, sd)

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

x_att = 'mobile_wt'
y_att = 'battery_power'
# plot_pairs = [('ram','pc'),('mobile_wt','battery_power'),('n_cores','ram')]
plot_pairs = [('Area','Perimeter')]

for x,y in plot_pairs:
    x_l = X[:,attributeName_to_index[x]]
    y_l = X[:,attributeName_to_index[y]]
    plt.title(str(x)+" vs. "+str(y))
    plt.scatter(x=x_l,
                y=y_l)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

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

