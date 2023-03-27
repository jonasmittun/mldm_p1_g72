import torch
from numpy import argmin

from regression_importdata import *
from sklearn import model_selection

K1 = 2
K2 = 2

baseline = y.mean()

# models = [lm.LinearRegression(fit_intercept=True), lm.LinearRegression(fit_intercept=False)] # Models

# models_control_parameters = [range()]


CVOuter = model_selection.KFold(n_splits=K1, shuffle=True)
CVInner = model_selection.KFold(n_splits=K2, shuffle=True)
i = 0

def validationError(X, y, model):
    distance_function = lambda yt, xt: (yt-model.predict(xt))**2
    tmp_sum = 0
    for i in range(len(y)):
        tmp_sum += distance_function(y[i],X[i,:])
    return tmp_sum/len(y)



sizeDval = np.zeros(K2)

Egen = 0
print("{}\t{}\t{}\t{}\t{}\t{}".format('fold i','h*','error','lambda*','error','Baseline error'))
for par_index, test_index in CVOuter.split(X):
    j = 0
    X_test, y_test = X[test_index, :], y[test_index]
    for train_index, val_index in CVInner.split(X[par_index,:]):
        sizeDval[j] = sum(val_index)
        X_train, y_train = X[train_index, :], y[train_index]
        X_val, y_val = X[val_index, :], y[val_index]
        for s,model in enumerate(models):
            # Train model
            fitted_model = model.fit(X_train, y_train)
            Eval[s,j] = np.square(y_val-fitted_model.predict(X_val)).sum()/y_val.shape[0]# validationError(X[val_index,:],y[val_index],fittedmodel) # Validation error

        j += 1
    # Select ANN* - For each s compute generalization error
    for s, _ in enumerate(hs):
        runningSum = 0
        for j in range(K2):
            runningSum += (sizeDval[j]/sum(par_index)) * Eval[s, j]
        EgenS[s] = runningSum
    # Select optimal model M*
    index_opt = argmin(EgenS)
    h_opt = hs[index_opt]

    # Baseline
    baseline = y_par.mean()
    Error_test_baseline = np.square(y_test - baseline).sum(axis=0)/y_test.shape[0]

    # RLR - Here the optimal value of lambda is found.
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_intercept[par_index, :], y[par_index], lambdas, K2)
    index_lambda_opt = np.where(lambdas == opt_lambda)
    rlr_opt = mean_w_vs_lambda[:, index_lambda_opt]

    # Train ANN on D_Par
    net, _, _ = train_neural_net(ann_model(h_opt), loss_fn, X=torch.Tensor(X_par), y=torch.Tensor(y_par).unsqueeze(1), n_replicates=1, max_iter=max_iter)

    y_val_est = net(X_val).squeeze()
    se = (y_val_est.float() - y_val.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_val)).data.numpy()  # mean
    Error_test_ann = mse

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
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr = np.linalg.solve(XtX+lambdaI, Xty).squeeze()
    # Make predictions on test data
    rlr_y_est = rlr_X_test @ w_rlr
    # Compute error
    Error_test_rlr = np.square(y_test - rlr_y_est).sum(axis=0)/y_test.shape[0]


    # Estimate generalization error on the fly
    # Egen += (sum(test_index)/N) * Etest_i
    print("{}\t{}\t{}\t{}\t{}\t{}".format(i, h_opt, Error_test_ann, opt_lambda, Error_test_rlr, Error_test_baseline))
    i += 1


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