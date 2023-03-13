
import sklearn.linear_model as lm
# Cross validation for the classification models
import numpy as np
from matplotlib.pyplot import scatter, plot
from numpy import argmin

from regression_importdata import *

from sklearn import model_selection

K1 = 5
K2 = 5
models = [lm.LinearRegression(fit_intercept=True),lm.LinearRegression(fit_intercept=False)] # Models
# models_control_parameters = [range()]
Eval = np.zeros((len(models),K2))
EgenS = np.zeros(len(models))
Etest = np.zeros(K1)


CVOuter = model_selection.KFold(n_splits=K1,shuffle=True)
CVInner = model_selection.KFold(n_splits=K2,shuffle=True)
i = 0


def validationError(X, y, model):
    distance_function = lambda yt, xt: (yt-model.predict(xt))**2
    tmp_sum = 0
    for i in range(len(y)):
        tmp_sum += distance_function(y[i],X[i,:])
    return tmp_sum/len(y)



sizeDval = np.zeros(K2)

Egen = 0
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
    # For each s compute generalization error
    for s, _ in enumerate(models):
        runningSum = 0
        for j in range(K2):
            runningSum += (sizeDval[j]/sum(par_index)) * Eval[s,j]
        EgenS[s] = runningSum
    # Select optimal model M*
    index_opt = argmin(EgenS)
    model_opt = models[index_opt]

    # Train model on X[par_index,:]
    fitted_model_opt = model_opt.fit(X[par_index,:],y[par_index])
    # Compute Etest_i
    Etest_i = np.square(y_test-fitted_model_opt.predict(X_test)).sum()/y_test.shape[0]#validationError(X[test_index,:],y[test_index], fitted_model_opt) # Something

    # Estimate generalization error on the fly
    Egen += (sum(test_index)/N) * Etest_i
    i += 1


y_hat = fitted_model_opt.predict(X)

fig = figure()
scatter(y_hat,y)
xlabel("Predicted values")
ylabel("True values")
show()

fig = figure()
residuals = y-y_hat
plot(residuals, '.')
xlabel("Observations")
ylabel("Residual (y - y_hat)")
show()

fig = figure()
hist(residuals, bins=41)
show()