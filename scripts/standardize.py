import numpy as np

def prepend_zero(matrix):
    return np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)

def standardize_X(known, test, intercept=False):
    test = test - np.ones((test.shape[0], 1)) * known.mean(0)
    # Check if any column has 0 std
    std = np.std(known, 0)
    b = np.argwhere(std == 0)
    std[np.reshape(b, (len(b)))] = 1
    complete = test * (1 / std)
    if intercept:
        return prepend_zero(complete)
    return complete


def standardize_train_test_pair(known, test, intercept=False):
    return standardize_X(known,known,intercept=intercept), standardize_X(known,test,intercept=intercept)


def standardize_y_for_training(target):
    target = target - np.ones(target.shape[0]) * np.mean(target)
    std = np.std(target)
    if std == 0:
        return 0
    return target * (1 / std)


def reverse_standardization_y(target, prediction):
    mean = np.mean(target)
    std = np.std(target)
    return prediction * std + mean


def find_optimal_parameter(params, e_val, split_sizes, K):
    e_gen = [0] * len(params)
    for s, _ in enumerate(params):
        tmp_sum = 0
        for j in range(K):
            tmp_sum += (split_sizes[j] / sum(split_sizes)) * e_val[s, j]
        e_gen[s] = tmp_sum
    # Select optimal parameter
    return params[np.argmin(e_gen)]
