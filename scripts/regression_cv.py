import numpy
import torch
from numpy import argmin
from toolbox_02450 import rlr_validate, train_neural_net
from regression_importdata import *
from sklearn import model_selection
import scipy.stats as st
from itertools import chain
from matplotlib.pyplot import (figure, subplot, xlabel, ylabel,
                               xticks, yticks, legend, show, hist, title,
                               subplots_adjust, scatter, savefig, suptitle, plot, xlim, ylim)

# Parameters
K1 = 10  # Outer fold
K2 = 10  # Inner fold
lambdas = np.power(10., np.arange(-3, 2, 0.1))  # Regularization factors in RLR
hs = [i for i in range(1, 3)]  # Number of hidden units in ANN
NR = 1  # Number of iterations when training ANN
max_iter = 10000  # Maximum number of iterations when training ANN model.

# Old code - perhaps delete later
# models = [lm.LinearRegression(fit_intercept=True), lm.LinearRegression(fit_intercept=False)] # Models
# models_control_parameters = [range()]

# scale = 5
# def transform_y_to_scaled(target):
#     return np.power(np.array([scale]*target.shape[0]), target)
# def transform_scaled_to_y(scaled):
#     return np.emath.logn(scale,scaled)


# Functions that split the data for cross-validation
CVOuter = model_selection.KFold(n_splits=K1, shuffle=True)
CVInner = model_selection.KFold(n_splits=K2, shuffle=True)

i = 0  # Iterate of the outer loop

# Functions for computing validation error. Is not used so perhaps delete.
def computeMse(y,yhat):
    distance_function = lambda y1, y2: (y1-y2)**2
    tmp_sum = 0
    for i in range(len(y)):
        tmp_sum += distance_function(y[i],yhat[i])
    return tmp_sum/len(y)

# TODO: Handle the following bugs...
#  <ipython-input-2-b6e01f0b0fcb>:42: RuntimeWarning: divide by zero encountered in true_divide
#   return matrix * (1 / np.std(matrix, 0))
#  <ipython-input-2-b6e01f0b0fcb>:42: RuntimeWarning: invalid value encountered in multiply
#   return matrix * (1 / np.std(matrix, 0))
def standardize_X(matrix):
    matrix = matrix - np.ones((matrix.shape[0], 1)) * matrix.mean(0)
    return matrix * (1 / np.std(matrix, 0))

# ANN model
def ann_model(n_hidden_units):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hidden units
        # 1st transfer function, either Tanh or ReLU:
        torch.nn.Tanh(),
        # torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, 1)  # H hidden units to 1 output neuron
        # torch.nn.Sigmoid()  # final tranfer function
    )

def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred) / y_true) * 100

loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.HuberLoss(reduction='mean', delta=3.0)
loss_fn = mean_absolute_percentage_error


# Initialization of some variables used later
Eval = np.zeros((len(hs), K2))
EgenS = np.zeros(len(hs))
sizeDval = np.zeros(K2)
Egen = 0
ANN_loss = []
RLR_loss = []
BASE_loss = []


# Header of the table that we want to produce.
print("{}\t{}\t{}\t{}\t{}\t{}".format('fold i','h*','error','lambda*','error','Baseline error'))



# summaries, summaries_axes = plot.subplots(1,2, figsize=(10,5))
# # Make a list for storing assigned color of learning curve for up to K=10
# color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
#               'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

est_arr = list()
ANN_y_hat = np.empty((0))
RLR_y_hat = np.empty((0))
y_true = np.empty((0))
# ANN_y_hat = np.concatenate((ANN_y_hat, [4,5]))


# Outer loop
for par_index, test_index in CVOuter.split(X):
    j = 0  # Iterate for the inner loop.

    # Split data into test data and parameter data.
    X_test, y_test = X[test_index, :], y[test_index]
    X_par, y_par = X[par_index, :], y[par_index]

    # This inner loop cross-validates the ANN model (the second level cross-validation)
    for train_index, val_index in CVInner.split(X[par_index, :]):
        # Split parameter data into training data and validation data

        X_train, y_train = X_par[train_index, :], y_par[train_index]
        X_val, y_val = X_par[val_index, :], y_par[val_index]

        X_train = standardize_X(X_train)
        X_val = standardize_X(X_val)

        sizeDval[j] = len(val_index)
        # X_train_t, y_train_t = torch.Tensor(X_train), torch.Tensor(y_train)
        # X_val_t, y_val = torch.Tensor(X_val), y_val#, torch.Tensor(y_val)

        # Loop over the complexity parameter for the ANN model (number of hidden units)
        for s, h in enumerate(hs):
            # Train model ###### TODO Use pytorch directly
            net, final_loss, learning_curve = train_neural_net(ann_model(h),
                                                               loss_fn,
                                                               X=torch.Tensor(X_train),
                                                               y=torch.Tensor(y_train).unsqueeze(1),
                                                               n_replicates=NR,
                                                               max_iter=max_iter)
            # y_pred = net(X_val).float()   # activation of final note, i.e. prediction of network
            # y_val = y_val.float()

            # Validate the model on the validation data and store results
            y_val_est_tensor = net(torch.Tensor(X_val)).squeeze()
            y_val_est = y_val_est_tensor.detach().numpy()
            est_arr.append(y_val_est)
            se = np.square(y_val - y_val_est) # squared error
            mse = np.mean(se)
            Eval[s, j] = mse

        j += 1
    # End of inner loop

    # Select ANN* - For each complexity parameter: compute generalization error
    for s, _ in enumerate(hs):
        runningSum = 0
        for j in range(K2):
            runningSum += (sizeDval[j]/len(par_index)) * Eval[s, j]
        EgenS[s] = runningSum
    # Select optimal model M*
    index_opt = argmin(EgenS)
    h_opt = hs[index_opt]  # The optimal number of hidden units in this outer fold.

    # RLR - Here the optimal value of lambda is found.
    # This is essentially another inner loop, but the function rlr_validate does this.

    # Change data to have intercept attribute for the RLR model

    X_intercept = np.concatenate((np.ones((X.shape[0], 1)), standardize_X(X)), 1)
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_intercept[par_index, :], y[par_index], lambdas, K2)

    # Pick out the weights for the optimal lambda
    index_lambda_opt = np.where(lambdas == opt_lambda)
    rlr_opt = mean_w_vs_lambda[:, index_lambda_opt]  # The optimal coefficients for the RLR model.

    ##### With all three models selected, we then train them on the parameter data and test on the test data.

    # Baseline model (computing the average and use this as prediction)
    baseline = y_par.mean()
    # Compute performance of the baseline model
    baseline_loss = np.square(y_test - baseline)
    Error_test_baseline = np.square(y_test - baseline).sum(axis=0)/y_test.shape[0]

    # Train ANN on D_Par
    net, _, _ = train_neural_net(ann_model(h_opt), loss_fn, X=torch.Tensor(X_par),
                                 y=torch.Tensor(y_par).unsqueeze(1), n_replicates=NR,
                                 max_iter=max_iter)
    # Compute generalization error of ANN
    y_test_est_tensor = net(torch.Tensor(X_test)).squeeze()
    y_test_est = y_test_est_tensor.detach().numpy()
    ANN_y_hat = numpy.concatenate((ANN_y_hat, y_test_est))
    se = np.square(y_test - y_test_est)  # squared error
    mse = np.mean(se)
    Error_test_ann = mse
    y_true = np.concatenate((y_true, y_test))

    fig = figure()
    plot(learning_curve)
    show()

    # Train RLR on D_par:
    # Set up variables in order to train the RLR model
    rlr_X_test = X_intercept[test_index, :]
    rlr_X_par = X_intercept[par_index, :]
    mu = np.mean(rlr_X_par[:, 1:], 0)
    sigma = np.std(rlr_X_par[:, 1:], 0)
    rlr_X_par[:, 1:] = (rlr_X_par[:, 1:] - mu) / sigma
    rlr_X_test[:, 1:] = (rlr_X_test[:, 1:] - mu) / sigma
    Xty = rlr_X_par.T @ y_par
    XtX = rlr_X_par.T @ rlr_X_par
    # Train the RLR model on X[par_index, :]
    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr = np.linalg.solve(XtX+lambdaI, Xty).squeeze()
    # Make predictions on test data
    rlr_y_est = rlr_X_test @ w_rlr
    RLR_y_hat = np.concatenate((RLR_y_hat,rlr_y_est))
    # Compute error
    rlr_loss = np.square(y_test - rlr_y_est)
    Error_test_rlr = np.square(y_test - rlr_y_est).sum(axis=0)/y_test.shape[0]

    # Store all losses such that we can do statistics
    ANN_loss += se.tolist()
    RLR_loss += rlr_loss.tolist()
    BASE_loss += baseline_loss.tolist()

    # Estimate generalization error on the fly
    # Egen += (sum(test_index)/N) * Etest_i

    # Print statement to make a table (formatted just to copy into LaTeX)
    print("{:d} & {:d} & {:.10f} & {:f} & {:.10f} & {:.10f} \\\ \hline".format(i+1, h_opt, Error_test_ann, opt_lambda, Error_test_rlr, Error_test_baseline))
    i += 1
# End of outer loop.

# Losses of the three models. Make into numpy arrays.
ANN_loss = np.array(ANN_loss)
RLR_loss = np.array(RLR_loss)
BASE_loss = np.array(BASE_loss)

# Do statistics
alpha = 0.05

# Compare model A and B
CIAB = st.t.interval(1-alpha, len(ANN_loss)-1, loc=np.mean(ANN_loss - RLR_loss), scale=st.sem(ANN_loss - RLR_loss))  # Confidence interval
pAB = 2*st.t.cdf(-np.abs(np.mean(ANN_loss - RLR_loss))/st.sem(ANN_loss - RLR_loss), df=len(ANN_loss)-1)  # p-value

# Compare model A and C
CIAC = st.t.interval(1-alpha, len(ANN_loss)-1, loc=np.mean(ANN_loss - BASE_loss), scale=st.sem(ANN_loss - BASE_loss))  # Confidence interval
pAC = 2*st.t.cdf(-np.abs(np.mean(ANN_loss - BASE_loss))/st.sem(ANN_loss - BASE_loss), df=len(ANN_loss)-1)  # p-value

# Compare model B and C
CIBC = st.t.interval(1-alpha, len(RLR_loss)-1, loc=np.mean(RLR_loss - BASE_loss), scale=st.sem(RLR_loss - BASE_loss))  # Confidence interval
pBC = 2*st.t.cdf(-np.abs(np.mean(RLR_loss - BASE_loss))/st.sem(RLR_loss - BASE_loss), df=len(RLR_loss)-1)  # p-value

print("The ANN-RLR confidence interval:", CIAB)
print("The ANN-RLR p-value:", pAB)
print("The ANN-BASE confidence interval:", CIAC)
print("The ANN-BASE p-value:", pAC)
print("The RLR-BASE confidence interval:", CIBC)
print("The RLR-BASE p-value:", pBC)

# y_hat = fitted_model_opt.predict(X)

# y_true format up to this point is list of lists, which isn't compatible with scatterplot
y_true = list(chain(*y_true))

fig = figure()
scatter(y_true,ANN_y_hat)
plot(xlim(), xlim(), c='black')
legend(["","y=x"])
xlabel("True values")
ylabel("Predicted values")
title("ANN vs. True value")
savefig("../plots/ANNvsTrue.svg", bbox_inches='tight')
show()

fig = figure()
scatter(y_true,RLR_y_hat)
plot(xlim(), xlim(), c='black')
legend(["","y=x"])
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