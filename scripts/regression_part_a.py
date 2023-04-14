# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid, plot)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from regression_importdata import *
from toolbox_02450 import rlr_validate, train_neural_net
import torch
def standardize_X(matrix):
    matrix = matrix - np.ones((matrix.shape[0], 1)) * matrix.mean(0)
    return matrix * (1 / np.std(matrix, 0))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), standardize_X(X)), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10

# Values of lambda
lambdas = np.power(10., np.arange(-3, 2, 0.1))

# Initialize variables
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, K)

figure(figsize=(10, 5))
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
xlabel('Regularization factor')
ylabel('Squared error (cross-validation)')
legend(['Train error', 'Validation error'])
grid()
show()
print(opt_val_err)
index_of_opt_lambda = np.where(lambdas == opt_lambda)
coef = mean_w_vs_lambda[:,index_of_opt_lambda]
print(coef)
# Display results
# print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
# print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format(
#     (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
# print(
#     '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))
#
# print('Weights in last fold:')
# for m in range(M):
#     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, -1], 2)))




# Create crossvalidation partition for evaluation
K = 5

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




# ANN model
def ann_model(n_hidden_units):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hidden units
        # 1st transfer function, either Tanh or ReLU:
        #torch.nn.Tanh(),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, 1)  # H hidden units to 1 output neuron
        # torch.nn.Sigmoid()  # final tranfer function
    )
loss_fn = torch.nn.MSELoss()


CV = model_selection.KFold(n_splits=K, shuffle=True)
max_iterations = 10000
ks = [i for i in range(1, 8)]
sizeDval_KNN = np.zeros(K)
sizeDval_KNN_train = np.zeros(K)
j = 0
Eval_KNN = np.zeros((len(ks), K))
Eval_KNN_train = np.zeros((len(ks), K))
for ind_train, ind_test in CV.split(X):
    X_train, y_train = X[ind_train, :], y[ind_train]
    X_test, y_test = X[ind_test, :], y[ind_test]
    sizeDval_KNN[j] = len(ind_test)
    sizeDval_KNN_train[j] = len(ind_train)
    for s, h in enumerate(ks):
        net, final_loss, learning_curve = train_neural_net(ann_model(h),
                                                           loss_fn,
                                                           X=torch.Tensor(X_train),
                                                           y=torch.Tensor(y_train).unsqueeze(1),
                                                           n_replicates=1,
                                                           max_iter=10000)

        y_test_est_tensor = net(torch.Tensor(torch.Tensor(X_test))).squeeze()
        y_test_est = y_test_est_tensor.detach().numpy()
        Eval_KNN[s, j] = np.mean(np.square(y_test - y_test_est))
        y_train_est_tensor = net(torch.Tensor(torch.Tensor(X_train))).squeeze()
        y_train_est = y_train_est_tensor.detach().numpy()
        Eval_KNN_train[s, j] = np.mean(np.square(y_train - y_train_est))
    j += 1


k_opt, knn_validation_error, knn_training_error = compute_errors_and_opt(ks, Eval_KNN,Eval_KNN_train,sizeDval_KNN,sizeDval_KNN_train)


figure(figsize=(10, 5))
title('Optimal k: {}'.format(k_opt))
plot(ks, knn_validation_error, 'b.-', knn_training_error, 'r.-')
xlabel('Number of hidden units')
ylabel('Squared mean error (cross-validation)')
legend(['Validation error','Training error'])
grid()
show()
print("The optimal h is", k_opt)


print('Ran regression part a >:)')
