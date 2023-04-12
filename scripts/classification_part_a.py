# exercise 8.1.1
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid, plot, xscale)
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm
from sklearn import model_selection
from classification_importdata import *

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10

def compute_errors_and_opt(complexity, eval_error, train_error, size_val, size_train):
    EgenS = np.zeros(len(complexity))
    EgenS_train = np.zeros(len(complexity))
    for s, _ in enumerate(complexity):
        runningSum = 0
        runningSum_train = 0
        for k in range(K):
            runningSum += (size_val[k] / N) * eval_error[s, k]
            runningSum_train += (size_train[k] / sum(size_train)) * train_error[s, k]
        EgenS[s] = runningSum
        EgenS_train[s] = runningSum_train
    index_opt = np.argmin(EgenS)
    complexity_opt = complexity[index_opt]  # The optimal number of neighbors
    return complexity_opt, EgenS, EgenS_train


CV = model_selection.KFold(n_splits=K, shuffle=True)
# Values of lambda
lambdas = np.power(10., np.arange(-3, 2, 0.05))
# lambdas = np.arange(10**(-3), 2, 0.05)
max_iterations = 10000
K_max = 20
ks = [i for i in range(1, K_max + 1)]
sizeDval_KNN = np.zeros(K)
sizeDval_KNN_train = np.zeros(K)
dist = 1
metric = "minkowski"
metric_params = None
j = 0
Eval_KNN = np.zeros((len(ks), K))
Eval_KNN_train = np.zeros((len(ks), K))
for ind_train, ind_test in CV.split(X):
    X_train, y_train = X[ind_train, :], y[ind_train]
    X_test, y_test = X[ind_test, :], y[ind_test]
    sizeDval_KNN[j] = len(ind_test)
    sizeDval_KNN_train[j] = len(ind_train)
    for s, k_nearest in enumerate(ks):
        knclassifier = KNeighborsClassifier(n_neighbors=k_nearest, p=dist,
                                            metric=metric,
                                            metric_params=metric_params)
        knclassifier.fit(X_train, y_train)
        y_val_est = knclassifier.predict(X_test)
        y_train_est = knclassifier.predict(X_train)

        Eval_KNN[s, j] = np.sum(y_val_est != y_test) / len(y_test)
        Eval_KNN_train[s, j] = np.sum(y_train_est != y_train) / len(y_train)
    j += 1


k_opt, knn_validation_error, knn_training_error = compute_errors_and_opt(ks, Eval_KNN,Eval_KNN_train,sizeDval_KNN,sizeDval_KNN_train)


figure(figsize=(10, 5))
title('Optimal k: {}'.format(k_opt))
plot(ks, knn_validation_error, 'r.-', knn_training_error, 'b.-')
xlabel('Regularization factor')
ylabel('Squared error (cross-validation)')
legend(['Training error','Validation error'])
grid()
show()
print("The optimal k is", k_opt)

# RMR
j = 0
Eval_KNN = np.zeros((len(lambdas), K))
Eval_KNN_train = np.zeros((len(lambdas), K))
for ind_train, ind_test in CV.split(X):
    X_train, y_train = X[ind_train, :], y[ind_train]
    X_test, y_test = X[ind_test, :], y[ind_test]
    sizeDval_KNN[j] = len(ind_test)
    sizeDval_KNN_train[j] = len(ind_train)
    for s, regularization in enumerate(lambdas):
        knclassifier = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                             tol=1e-4, random_state=1,
                                             penalty='l2', C=1 / regularization,
                                             max_iter=max_iterations)
        knclassifier.fit(X_train, y_train)
        y_val_est = knclassifier.predict(X_test)
        y_train_est = knclassifier.predict(X_train)

        Eval_KNN[s, j] = np.sum(y_val_est != y_test) / len(y_test)
        Eval_KNN_train[s, j] = np.sum(y_train_est != y_train) / len(y_train)

    j += 1



lambda_opt, validation_error, train_error = compute_errors_and_opt(lambdas, Eval_KNN, Eval_KNN_train, sizeDval_KNN, sizeDval_KNN_train)


figure(figsize=(10, 5))
title('Optimal lambda: 1e{0}'.format(lambda_opt))
plot(lambdas, train_error, 'r.-', validation_error, 'b.-')
xscale('log')
xlabel('Regularization factor')
ylabel('Squared error (cross-validation)')
legend(['Train error','Validation error'])
grid()
show()
print("The optimal lambda is", lambda_opt)

print('Ran classification part a >:)')
