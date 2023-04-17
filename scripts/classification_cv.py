import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy import argmin
from classification_importdata import *
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from toolbox_02450 import mcnemar
from matplotlib.pyplot import (figure, subplot, xlabel, ylabel,
                               xticks, yticks, legend, show, hist, title,
                               subplots_adjust, scatter, savefig, suptitle, subplots)
from standardize import *

# Parameters
K1 = 2  # Outer fold
K2 = 2  # Inner fold
lambdas = np.power(10., np.arange(-4, 3, 0.05))  # Regularization factors in Multinomial Regression
max_iterations = 10000
K_max = 20
ks = [i for i in range(1, K_max + 1)]

# Functions that split the data for cross-validation
CVOuter = model_selection.KFold(n_splits=K1, shuffle=True)
CVInner = model_selection.KFold(n_splits=K2, shuffle=True)

# Initialization of some variables used later
KNN_hats = []
RMR_hats = []
BASE_hats = []
Y_TRUE = []

# Header of the table that we want to produce.
print("{}\t{}\t{}\t{}\t{}\t{}".format('fold i', 'k*', 'error', 'lambda*', 'error', 'Baseline error'))

# KNN parameters
dist = 2
metric = 'minkowski'
metric_params = {}


# metric='mahalanobis'
# metric_params={'V': cov(X_train, rowvar=False)}


def inner_knn(X_par, y_par, params, cv_k):
    e_val = np.zeros((len(params), cv_k))
    split_sizes = np.zeros(cv_k)
    j = 0  # KNN inner loop
    for train_index, val_index in CVInner.split(X_par):

        # extract training and test set for current CV fold
        X_train, X_val = standardize_train_test_pair(X_par[train_index], X_par[val_index])
        y_train, y_val = y_par[train_index], y_par[val_index]

        split_sizes[j] = len(val_index)

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for i, k_i in enumerate(params):
            knclassifier = KNeighborsClassifier(n_neighbors=k_i, p=dist,
                                                metric=metric,
                                                metric_params=metric_params)
            knclassifier.fit(X_train, y_train)
            y_val_est = knclassifier.predict(X_val)
            val_error_rate = np.sum(y_val_est != y_val) / len(y_val)
            e_val[i, j] = val_error_rate

        j += 1

    return find_optimal_parameter(ks, e_val, split_sizes, cv_k)


def inner_rmr(X_par, y_par, params, cv_k):
    e_val = np.zeros((len(params), cv_k))
    split_sizes = np.zeros(cv_k)
    j = 0  # KNN inner loop
    for train_index, val_index in CVInner.split(X_par):

        # extract training and test set for current CV fold
        X_train, X_val = standardize_train_test_pair(X_par[train_index], X_par[val_index])
        y_train, y_val = y_par[train_index], y_par[val_index]

        split_sizes[j] = len(val_index)

        # Loop over the complexity parameter for the ANN model (number of hidden units)
        for i, lambda_i in enumerate(lambdas):
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                        tol=1e-4, random_state=1,
                                        penalty='l2', C=1 / lambda_i,
                                        max_iter=max_iterations)
            mdl.fit(X_train, y_train)
            coe = mdl.coef_
            inter = mdl.intercept_
            y_val_est = mdl.predict(X_val)

            val_error_rate = np.sum(y_val_est != y_val) / len(y_val)
            e_val[i, j] = val_error_rate

        j += 1

    # Select MRM* - For each complexity parameter: compute generalization error

    return find_optimal_parameter(lambdas, e_val, split_sizes, cv_k)


# Outer loop
i = 0  # Iterate of the outer loop
for par_index, test_index in CVOuter.split(X):
    # Split data into test data and parameter data.
    X_test, y_test = X[test_index, :], y[test_index]
    X_par, y_par = X[par_index, :], y[par_index]

    # KNN Classifier ###############################################################
    k_opt = inner_knn(X_par, y_par, ks, K2)

    X_par_knn, X_test_knn = standardize_train_test_pair(X_par, X_test)
    knclassifier = KNeighborsClassifier(n_neighbors=k_opt, p=dist,
                                        metric=metric,
                                        metric_params=metric_params)
    knclassifier.fit(X_par_knn, y_par)
    y_hat_knn = knclassifier.predict(X_test_knn)
    knn_loss = y_hat_knn != y_test
    Error_test_KNN = np.sum(knn_loss) / len(y_test)
    # ###############################################################################

    #### Regularized Multinomial Regression (Multi-class logistic regression)
    lambda_opt = inner_rmr(X_par, y_par, lambdas, K2)

    X_par_rmr, X_test_rmr = standardize_train_test_pair(X_par, X_test, intercept=True)
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                tol=1e-4, random_state=1,
                                penalty='l2', C=1 / lambda_opt,
                                max_iter=max_iterations)
    mdl.fit(X_par_rmr, y_par)
    y_hat_rmr = mdl.predict(X_test_rmr)
    rmr_loss = y_hat_rmr != y_test
    Error_test_RMR = np.sum(rmr_loss) / len(y_test)
    #################################################################################

    # Baseline model (Computing the class with most occurrences in the data and use this as prediction)
    baseline = np.argmax(np.array([np.sum(y_par == 0), np.sum(y_par == 1), np.sum(y_par == 2)]))
    # Compute performance of the baseline model
    y_hat_base = np.ones(len(y_test)) * baseline
    baseline_loss = y_test != baseline
    Error_test_baseline = np.sum(baseline_loss) / len(y_test)
    ################################################################################

    # Store all predictions such that we can do statistics
    KNN_hats.append(y_hat_knn)
    RMR_hats.append(y_hat_rmr)
    BASE_hats.append(y_hat_base)
    Y_TRUE.append(y_test)

    # Print statement to make a table (formatted just to copy into LaTeX)
    print("{:d} & {:d} & {:.5f} & {:f} & {:.5f} & {:.5f} \\\ \hline".format(i + 1, k_opt, Error_test_KNN, lambda_opt,
                                                                            Error_test_RMR, Error_test_baseline))
    i += 1
# End of outer loop.

# Predictions of the three models. Make into numpy arrays.
KNN_hats = np.concatenate(KNN_hats)
RMR_hats = np.concatenate(RMR_hats)
BASE_hats = np.concatenate(BASE_hats)
Y_TRUE = np.concatenate(Y_TRUE)

# Do statistics
alpha = 0.05
# Compare model A and B (KNN against RMR)
[thetahatAB, CIAB, pAB] = mcnemar(Y_TRUE, KNN_hats, RMR_hats, alpha=alpha)
# Compare model A and C (KNN against BASE)
[thetahatAC, CIAC, pAC] = mcnemar(Y_TRUE, KNN_hats, BASE_hats, alpha=alpha)
# Compare model B and C (RMR against BASE)
[thetahatBC, CIBC, pBC] = mcnemar(Y_TRUE, RMR_hats, BASE_hats, alpha=alpha)


# Plot KNN class labels
# Matrix scatter plot of attributes
def matrix_scatter_plot(attributes, prediction, modelname):
    r_mask = np.ones(N, dtype=bool)
    figure(figsize=(12, 10))
    m = len(attributes)
    for i, m1 in enumerate(attributes):
        for j, m2 in enumerate(attributes):
            subplot(m, m, i * m + j + 1)
            for c in range(C):
                class_mask = ((prediction == c) & r_mask)
                scatter(np.array(X_original[class_mask, m2]), np.array(X_original[class_mask, m1]), marker='.',
                        s=8 * (10 - m), c=class_colors[c], alpha=0.50)
                if i == m - 1:
                    xticks(fontsize=7)
                    xlabel(attributeNames[m2])
                else:
                    xticks([])
                if j == 0:
                    yticks(fontsize=7)
                    ylabel(attributeNames[m1])
                else:
                    yticks([])
    subplot(m, m, 1).legend(classNames, bbox_to_anchor=(2, 2.3), loc='upper right')
    suptitle('Attribute scatter plot matrix', fontsize=14)
    savefig("../plots/spm-{}-est.svg".format(modelname), bbox_inches='tight')



att = [True for _ in range(M)]
# att = [False,False,True,True,False,False,False,False,False]
all_att = np.arange(M)
masked = all_att[att]
matrix_scatter_plot(masked, y, "true")
matrix_scatter_plot(masked, KNN_hats, "knn")
matrix_scatter_plot(masked, RMR_hats, "rmr")