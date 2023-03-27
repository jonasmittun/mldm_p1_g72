# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from regression_importdata import *
from toolbox_02450 import rlr_validate


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
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

print('Ran regression part a >:)')
