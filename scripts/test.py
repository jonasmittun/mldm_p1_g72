# Test of sklearns GridSearchCV for 2-layer cross validation
# Currently not working :clown:
# Source on discord - to be added later

# automatic nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from importdata_csv import *

# create dataset
# X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
# configure the cross-validation procedure
cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# define the model
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
# define search space
# params = dict()
# params['normalize'] = [True, False]
params = [{'n_features_to_select': list(range(2, 10))}]
# space['n_estimators'] = [10, 100, 500]
# space['max_features'] = [2, 4, 6]
# define search
search = GridSearchCV(model, params, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
# configure the cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# execute the nested cross-validation
scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
