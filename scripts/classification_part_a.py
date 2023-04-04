# exercise 8.1.1
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid, plot)
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from sklearn import model_selection
from classification_importdata import *

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)
# Values of lambda
lambdas = np.power(10., np.arange(-3, 2, 0.05))
max_iterations = 10000
K_max = 20
ks = [i for i in range(1, K_max + 1)]
sizeDval_KNN = np.zeros(K)
dist = 1
metric = "minkowski"
metric_params = None
j = 0
Eval_KNN = np.zeros((len(ks), K))
for ind_train, ind_test in CV.split(X):
    X_train, y_train = X[ind_train, :], y[ind_train]
    X_test, y_test = X[ind_test, :], y[ind_test]
    sizeDval_KNN[j] = sum(ind_test)
    for s, k_nearest in enumerate(ks):
        knclassifier = KNeighborsClassifier(n_neighbors=k_nearest, p=dist,
                                            metric=metric,
                                            metric_params=metric_params)
        knclassifier.fit(X_train, y_train)
        y_val_est = knclassifier.predict(X_test)
        # errors[j, k_nearest - 1] = np.sum(y_val_est[0] != y_val[0])
        # dy.append(y_val_est)

        val_error_rate = np.sum(y_val_est != y_test) / len(y_test)
        Eval_KNN[s, j] = val_error_rate

    # dy = np.stack(dy, axis=1)
    # yhat.append(dy)
    # y_true.append(y_val)
    j += 1

EgenS = np.zeros(len(ks))
for s, _ in enumerate(ks):
    runningSum = 0
    for k in range(K):
        runningSum += (sizeDval_KNN[k] / N) * Eval_KNN[s, k]
    EgenS[s] = runningSum
index_opt = np.argmin(EgenS)
k_opt = ks[index_opt]  # The optimal number of neighbors

figure(figsize=(10, 5))
title('Optimal k: {}'.format(k_opt))
plot(ks, EgenS, 'b.-')
xlabel('Regularization factor')
ylabel('Squared error (cross-validation)')
legend(['Train error', 'Validation error'])
grid()
show()
print("The optimal k is", k_opt)

# RMR
j = 0
Eval_KNN = np.zeros((len(lambdas), K))
for ind_train, ind_test in CV.split(X):
    X_train, y_train = X[ind_train, :], y[ind_train]
    X_test, y_test = X[ind_test, :], y[ind_test]
    sizeDval_KNN[j] = sum(ind_test)
    for s, regularization in enumerate(lambdas):
        knclassifier = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                             tol=1e-4, random_state=1,
                                             penalty='l2', C=1 / regularization,
                                             max_iter=max_iterations)
        knclassifier.fit(X_train, y_train)
        y_val_est = knclassifier.predict(X_test)
        # errors[j, k_nearest - 1] = np.sum(y_val_est[0] != y_val[0])
        # dy.append(y_val_est)

        val_error_rate = np.sum(y_val_est != y_test) / len(y_test)
        Eval_KNN[s, j] = val_error_rate

    # dy = np.stack(dy, axis=1)
    # yhat.append(dy)
    # y_true.append(y_val)
    j += 1

EgenS = np.zeros(len(lambdas))
for s, _ in enumerate(lambdas):
    runningSum = 0
    for k in range(K):
        runningSum += (sizeDval_KNN[k] / N) * Eval_KNN[s, k]
    EgenS[s] = runningSum
index_opt = np.argmin(EgenS)
lambda_opt = lambdas[index_opt]  # The optimal number of neighbors

figure(figsize=(10, 5))
title('Optimal lambda: 1e{0}'.format(lambda_opt))
loglog(lambdas, EgenS, 'b.-')
xlabel('Regularization factor')
ylabel('Squared error (cross-validation)')
legend(['Train error', 'Validation error'])
grid()
show()
print("The optimal lambda is", lambda_opt)

print('Ran classification part a >:)')
