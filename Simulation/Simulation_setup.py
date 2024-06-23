import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# Simulate N d-dimensional feature vectors
def generate_feature_vectors(d, N):
    mean = np.zeros(d)
    cov_matrix = np.eye(d)  # using identity matrix (so assume independency between features)
    X = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=N)
    return X


# Generate potential outcomes
def potential_outcomes(X, mu0, mu1):
    Y0 = []
    Y1 = []
    for sample in X:
        err_0 = np.random.normal(loc=0, scale=1)
        err_1 = np.random.normal(loc=0, scale=1)
        Yi_0 = mu0(sample) + err_0
        Yi_1 = mu1(sample) + err_1
        Y0.append(Yi_0)
        Y1.append(Yi_1)
    return Y0, Y1


# Simulate data for a single experiment
def generate_data(N, e, d, mu0, mu1):
    X = generate_feature_vectors(d, N)
    Y0, Y1 = potential_outcomes(X, mu0, mu1)
    T = np.random.binomial(1, e(X), size=N)
    Y = T * Y1 + (1 - T) * Y0
    return X, T, Y0, Y1, Y


def make_dataframe(samples, W, Y0, Y1, Y):
    df = pd.DataFrame(samples)
    df.columns = [f"X{i}" for i in range(1, samples.shape[1] + 1)]
    df['T'] = W
    df['Y0'] = Y0
    df['Y1'] = Y1
    df['Y'] = Y
    df['cate'] = df['Y1'] - df['Y0']

    X = [f"X{i}" for i in range(1, samples.shape[1] + 1)]
    return df, X


# Calulating the MSE
def MSE(test):
    test_cate = test['cate'].tolist()
    test_pred_cate = test['pred_cate'].tolist()
    return mean_squared_error(test_cate, test_pred_cate)
