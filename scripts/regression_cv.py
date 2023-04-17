import numpy
import torch
from toolbox_02450 import rlr_validate, train_neural_net
from regression_importdata import *
from sklearn import model_selection
import scipy.stats as st
from matplotlib.pyplot import (figure, subplot, xlabel, ylabel,
                               xticks, yticks, legend, show, hist, title,
                               subplots_adjust, scatter, savefig, suptitle, plot, xlim, ylim)
from standardize import *

# Parameters
K1 = 10  # Outer fold
K2 = 10  # Inner fold
lambdas = np.power(10., np.arange(-3, 2, 0.1))  # Regularization factors in RLR
hs = [i for i in range(1, 3)]  # Number of hidden units in ANN
max_iter = 10000  # Maximum number of iterations when training ANN model.

# Functions that split the data for cross-validation
CVOuter = model_selection.KFold(n_splits=K1, shuffle=True)
CVInner = model_selection.KFold(n_splits=K2, shuffle=True)

# Initialization of some variables used later
ANN_loss = []
RLR_loss = []
BASE_loss = []

# Header of the table that we want to produce.
print("{}\t{}\t{}\t{}\t{}\t{}".format('fold i', 'h*', 'error', 'lambda*', 'error', 'Baseline error'))

ANN_y_hat = np.empty((0))
RLR_y_hat = np.empty((0))
y_true = np.empty((0))




def ann_model(n_hidden_units):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hidden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),
        # torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, 1)  # H hidden units to 1 output neuron
    )


loss_fn = torch.nn.MSELoss()


def train_ann_and_predict(x_train, x_test, y_train, hidden, max_iterations=10000):
    """
    The model also contain a standardization step of the target attribute.
    """
    ann_net, _, _ = train_neural_net(ann_model(hidden),
                                     loss_fn,
                                     X=torch.Tensor(x_train),
                                     y=torch.Tensor(standardize_y_for_training(y_train)).unsqueeze(1),
                                     n_replicates=1,
                                     max_iter=max_iterations)

    # Validate the model on the validation data and store results
    y_val_est_before = ann_net(torch.Tensor(x_test)).squeeze().detach().numpy()
    return reverse_standardization_y(y_train, y_val_est_before)


def inner_ann(X_par, y_par, params):
    e_val = np.zeros((len(hs), K2))
    split_sizes_ann = np.zeros(K2)
    j = 0
    for train_index, val_index in CVInner.split(X_par):
        X_train, X_val = standardize_train_test_pair(X_par[train_index, :], X_par[val_index, :])
        y_train, y_val = y_par[train_index], y_par[val_index]

        split_sizes_ann[j] = len(val_index)

        # Loop over the complexity parameter for the ANN model (number of hidden units)
        for i, h_i in enumerate(params):
            y_val_pred = train_ann_and_predict(X_train, X_val, y_train, h_i)
            e_val[i, j] = np.mean(np.square(y_val - y_val_pred))

        j += 1

    # Select ANN* - For each complexity parameter: compute generalization error
    return find_optimal_parameter(hs, e_val, split_sizes_ann,K2)


def inner_rlr(X_par, y_par, params, cv_k):
    # Change data to have intercept attribute for the RLR model
    X_par_intercept = prepend_ones(X_par)

    # Let rlr_validate compute the inner split
    _, opt_lambda, mean_w_vs_lambda, _, _ = rlr_validate(X_par_intercept, y_par, params, cv_k)

    # Pick out the weights for the optimal lambda
    # index_lambda_opt = np.where(lambdas == opt_lambda)
    # opt_weights = mean_w_vs_lambda[:, index_lambda_opt]  # The optimal coefficients for the RLR model.
    return opt_lambda  # The optimal coefficients for the RLR model.



def print_latex_table_row(fold, h_opt, Error_test_ann, lambda_opt, Error_test_rlr, Error_test_baseline):
    print("{:d} & {:d} & {:.10f} & {:f} & {:.10f} & {:.10f} \\\\ \\hline".format(fold, h_opt, Error_test_ann,
                                                                               lambda_opt,
                                                                               Error_test_rlr, Error_test_baseline))

# Outer loop
i = 0
for par_index, test_index in CVOuter.split(X):
    # Split data into test data and parameter data.
    X_test, y_test = X[test_index, :], y[test_index]
    X_par, y_par = X[par_index, :], y[par_index]

    # This inner loop cross-validates the ANN model (the second level cross-validation)
    h_opt = inner_ann(X_par, y_par, hs)

    # RLR - Here the optimal value of lambda is found.
    lambda_opt = inner_rlr(X_par, y_par, lambdas, K2)

    # With all three models selected, we then train them on the parameter data and test on the test data.

    # Baseline model ###############################################################
    # (computing the average and use this as prediction)
    baseline = y_par.mean()
    # Compute performance of the baseline model
    baseline_loss = np.square(y_test - baseline)
    Error_test_baseline = np.square(y_test - baseline).sum(axis=0) / y_test.shape[0]
    # ##############################################################################

    # Train ANN on D_Par ############################################
    X_par_ann, X_test_ann = standardize_train_test_pair(X_par, X_test)
    y_test_est = train_ann_and_predict(X_par_ann, X_test_ann, y_par, h_opt)

    ANN_y_hat = numpy.concatenate((ANN_y_hat, y_test_est))
    se = np.square(y_test - y_test_est)
    Error_test_ann = np.mean(se)
    y_true = np.concatenate((y_true, y_test))
    # ###############################################################

    # Train RLR on D_par ######################################################
    # Set up variables in order to train the RLR model
    rlr_X_par, rlr_X_test = standardize_train_test_pair(X_par, X_test, intercept=True)
    Xty = rlr_X_par.T @ y_par
    XtX = rlr_X_par.T @ rlr_X_par
    # Train the RLR model on X_par
    lambdaI = lambda_opt * np.eye(M + 1)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Make predictions on test data
    rlr_y_est = rlr_X_test @ w_rlr
    RLR_y_hat = np.concatenate((RLR_y_hat, rlr_y_est))
    # Compute error
    rlr_loss = np.square(y_test - rlr_y_est)
    Error_test_rlr = np.square(y_test - rlr_y_est).sum(axis=0) / y_test.shape[0]
    # ##########################################################################

    # Store all losses for statistical evaluation
    ANN_loss += se.tolist()
    RLR_loss += rlr_loss.tolist()
    BASE_loss += baseline_loss.tolist()

    # Print statement to make a table (formatted just to copy into LaTeX)
    print_latex_table_row(i + 1, h_opt, Error_test_ann, lambda_opt, Error_test_rlr, Error_test_baseline)
    i += 1
# End of outer loop.

# Losses of the three models. Make into numpy arrays.
ANN_loss = np.array(ANN_loss)
RLR_loss = np.array(RLR_loss)
BASE_loss = np.array(BASE_loss)


def CI(alpha, lossA, lossB):
    ciAB = st.t.interval(1 - alpha, len(lossA) - 1, loc=np.mean(lossA - lossB), scale=st.sem(lossA - lossB))
    pAB = 2 * st.t.cdf(-np.abs(np.mean(lossA - lossB)) / st.sem(lossA - lossB), df=len(lossA) - 1)
    return ciAB, pAB


# Do statistics
alpha = 0.05

# Compare model A and B
CI_ANN_RLR, p_ANN_RLR = CI(alpha, ANN_loss, RLR_loss)
CI_ANN_BASE, p_ANN_BASE = CI(alpha, ANN_loss, BASE_loss)
CI_RLR_BASE, p_RLR_BASE = CI(alpha, RLR_loss, BASE_loss)

print("The ANN-RLR confidence interval:", CI_ANN_RLR)
print("The ANN-RLR p-value:", p_ANN_RLR)
print("The ANN-BASE confidence interval:", CI_ANN_BASE)
print("The ANN-BASE p-value:", p_ANN_BASE)
print("The RLR-BASE confidence interval:", CI_RLR_BASE)
print("The RLR-BASE p-value:", p_RLR_BASE)

# Comparing true values to the predicted values in the outer fold
fig = figure()
scatter(y_true, ANN_y_hat)
plot(xlim(), xlim(), c='black')
legend(["", "y=x"])
xlabel("True values")
ylabel("Predicted values")
title("ANN vs. True value")
savefig("../plots/ANNvsTrue.svg", bbox_inches='tight')
show()

fig = figure()
scatter(y_true, RLR_y_hat)
plot(xlim(), xlim(), c='black')
legend(["", "y=x"])
xlabel("True values")
ylabel("Predicted values")
title("RLR vs. True value")
savefig("../plots/RLRvsTrue.svg", bbox_inches='tight')
show()

fig = figure()
plot(ANN_y_hat, color='red')
plot(y_true, color='blue')
legend(["Prediction", "True Value"])
xlabel("Instance No.")
ylabel("Refractive Index")
title("ANN Predictions vs. True Value")
savefig("../plots/ANNcomp.svg", bbox_inches='tight')
show()
