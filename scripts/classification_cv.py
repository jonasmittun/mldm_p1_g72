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
                               subplots_adjust, scatter, savefig,suptitle)


# Parameters
K1 = 10  # Outer fold
K2 = 10  # Inner fold
lambdas = np.power(10., np.arange(-3, 2, 0.05))  # Regularization factors in Multinomial Regression
max_iterations = 20000
K_max = 20
ks = [i for i in range(1, K_max + 1)]

# Functions that split the data for cross-validation
CVOuter = model_selection.KFold(n_splits=K1, shuffle=True)
CVInner = model_selection.KFold(n_splits=K2, shuffle=True)

i = 0  # Iterate of the outer loop

# Arrays for KNN results
yhat = []
y_true = []
knn_errors = []


# Functions for computing validation error. Is not used so perhaps delete.
def validationError(X, y, model):
    distance_function = lambda yt, xt: (yt - model.predict(xt)) ** 2
    tmp_sum = 0
    for i in range(len(y)):
        tmp_sum += distance_function(y[i], X[i, :])
    return tmp_sum / len(y)


# Initialization of some variables used later
sizeDval = np.zeros(K2)
Egen = 0
KNN_hats = []
RMR_hats = []
BASE_hats = []
Y_TRUE = []

def standardize_X(matrix):
    matrix = matrix - np.ones((matrix.shape[0], 1)) * matrix.mean(0)
    return matrix * (1 / np.std(matrix, 0))

# Change data to have intercept attribute for the RLR model
X_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

# Header of the table that we want to produce.
print("{}\t{}\t{}\t{}\t{}\t{}".format('fold i', 'k*', 'error', 'lambda*', 'error', 'Baseline error'))

# Outer loop
for par_index, test_index in CVOuter.split(X):

    # Split data into test data and parameter data.
    X_test, y_test = X[test_index, :], y[test_index]
    X_par, y_par = X[par_index, :], y[par_index]

    ## KNN Classifier

    # Error container for this outer iteration
    errors = np.zeros((N, K_max))

    # Distance metric
    dist = 2
    metric = 'minkowski'
    metric_params = {}
    # metric='mahalanobis'
    # metric_params={'V': cov(X_train, rowvar=False)}
    Eval_KNN = np.zeros((len(ks), K2))

    j = 0  # KNN inner loop
    sizeDval_KNN = np.zeros(K2)
    Eval = np.zeros((len(ks), K2))
    for train_index, val_index in CVInner.split(X_par):

        # extract training and test set for current CV fold
        X_train = standardize_X(X_par[train_index, :])
        y_train = y_par[train_index]
        X_val = standardize_X(X_par[val_index, :])
        y_val = y_par[val_index]
        sizeDval_KNN[j] = len(val_index)

        # Array containing test performance for classifiers
        dy = []

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for s, k_nearest in enumerate(ks):
            knclassifier = KNeighborsClassifier(n_neighbors=k_nearest, p=dist,
                                                metric=metric,
                                                metric_params=metric_params)
            knclassifier.fit(X_train, y_train)
            y_val_est = knclassifier.predict(X_val)

            val_error_rate = np.sum(y_val_est != y_val) / len(y_val)
            Eval_KNN[s, j] = val_error_rate

        j += 1

    EgenS = np.zeros(len(ks))
    for s, _ in enumerate(ks):
        runningSum = 0
        for k in range(K2):
            runningSum += (sizeDval_KNN[k] / len(par_index)) * Eval_KNN[s, k]
        EgenS[s] = runningSum
    # Select optimal model M*
    index_opt = argmin(EgenS)
    k_opt = ks[index_opt]  # The optimal number of neighbors

    knclassifier = KNeighborsClassifier(n_neighbors=k_opt, p=dist,
                                        metric=metric,
                                        metric_params=metric_params)
    knclassifier.fit(standardize_X(X_par), y_par)
    y_hat_knn = knclassifier.predict(X_test)
    knn_loss = y_hat_knn != y_test
    Error_test_KNN = np.sum(knn_loss) / len(y_test)

    #### Regularized Multinomial Regression (Multi-class logistic regression)
    sizeDval_RMR = np.zeros(K2)
    Eval_RMR = np.zeros((len(lambdas), K2))
    k = 0  # RMR inner loop
    EgenS = np.zeros(len(lambdas))
    for train_index, val_index in CVInner.split(X_par):
        # Split parameter data into training data and validation data
        sizeDval_RMR[k] = len(val_index)
        X_train, y_train = standardize_X(X_par[train_index, :]), y_par[train_index]
        X_val, y_val = standardize_X(X_par[val_index, :]), y_par[val_index]

        # Loop over the complexity parameter for the ANN model (number of hidden units)
        for s, regularization in enumerate(lambdas):
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                        tol=1e-4, random_state=1,
                                        penalty='l2', C=1 / regularization,
                                        max_iter=max_iterations)
            mdl.fit(X_train, y_train)
            y_val_est = mdl.predict(X_val)

            val_error_rate = np.sum(y_val_est != y_val) / len(y_val)
            Eval_RMR[s, k] = val_error_rate

        k += 1

    # Select MRM* - For each complexity parameter: compute generalization error

    for s, _ in enumerate(lambdas):
        runningSum = 0
        for k in range(K2):
            runningSum += (sizeDval_RMR[k] / len(par_index)) * Eval_RMR[s, k]
        EgenS[s] = runningSum
    # Select optimal model M*
    index_opt = argmin(EgenS)
    lambda_opt = lambdas[index_opt]  # The optimal number of hidden units in this outer fold.

    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                tol=1e-4, random_state=1,
                                penalty='l2', C=1 / lambda_opt,
                                max_iter=max_iterations)
    mdl.fit(standardize_X(X_par), y_par)
    y_hat_rmr = mdl.predict(X_test)
    rmr_loss = y_hat_rmr != y_test
    Error_test_RMR = np.sum(rmr_loss) / len(y_test)

    # With all three models selected, we then train them on the parameter data and test on the test data.

    # Baseline model (Computing the class with most occurences in the data and use this as prediction)
    baseline = np.argmax(np.array([np.sum(y_par == 0), np.sum(y_par == 1), np.sum(y_par == 2)]))
    # Compute performance of the baseline model
    y_hat_base = np.ones(len(y_test)) * baseline
    baseline_loss = y_test != baseline
    Error_test_baseline = np.sum(baseline_loss) / len(y_test)

    # Train KNN on D_Par

    # Store all predictions such that we can do statistics
    KNN_hats.append(y_hat_knn)
    RMR_hats.append(y_hat_rmr)
    BASE_hats.append(y_hat_base)
    Y_TRUE.append(y_test)

    # Estimate generalization error on the fly
    # Egen += (sum(test_index)/N) * Etest_i

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
    m = sum(attributes)
    for m1 in range(m):
        for m2 in range(m):
            subplot(m, m, m1*m + m2 + 1)
            for c in range(C):
                class_mask = ((prediction == c) & r_mask)
                scatter(np.array(X_original[class_mask, m2]), np.array(X_original[class_mask, m1]), marker='.', s=8*(10-m), c=class_colors[c], alpha=0.50)
                if m1 == m-1:
                    xticks(fontsize=7)
                    xlabel(attributeNames[m2])
                else:
                    xticks([])
                if m2 == 0:
                    yticks(fontsize=7)
                    ylabel(attributeNames[m1])
                else:
                    yticks([])
    subplot(m, m, 1).legend(classNames, bbox_to_anchor=(2, 2.3), loc='upper right')
    suptitle('Attribute scatter plot matrix', fontsize=14)
    savefig("../plots/spm-{}-est.svg".format(modelname), bbox_inches='tight')
    # show()

# att = [1 for _ in range(M)]
att = [1,0,0,1,0,1,0,0,1]
all_att = np.arange(M)
masked = all_att[att]
matrix_scatter_plot(masked, y, "true")
matrix_scatter_plot(masked, KNN_hats, "knn")
matrix_scatter_plot(masked, RMR_hats, "rmr")
matrix_scatter_plot(masked, BASE_hats, "base")

# print("The A-B confidence interval:", CIAB)
# print("The A-B p-value:", pAB)
# print("The A-C confidence interval:", CIAC)
# print("The A-C p-value:", pAC)
# print("The B-C confidence interval:", CIBC)
# print("The B-C p-value:", pBC)

# y_hat = fitted_model_opt.predict(X)
#
# fig = figure()
# scatter(y_hat,y)
# xlabel("Predicted values")
# ylabel("True values")
# show()
#
# fig = figure()
# residuals = y-y_hat
# plot(residuals, '.')
# xlabel("Observations")
# ylabel("Residual (y - y_hat)")
# show()
#
# fig = figure()
# hist(residuals, bins=41)
# show()
