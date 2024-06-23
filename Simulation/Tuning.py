import tensorflow as tf
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def tune_lgbm(X, y):
    num_leaves = 30            # [30, 50, 70] Controls complexity, should be smaller than 2^(max_depth)
    max_depth = 5               # [5, 7, 9] Constrain tree depth to prevent overfitting
    min_child_samples = 2       # [2, 5, 10]
    min_data_in_leaf = 20      # [20, 50, 100] Important parameter to prevent overfitting, setting it larger van
                                # avoid growing too deep a tree
    lgbm = LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, min_child_samples=min_child_samples,
                         min_data_in_leaf=min_data_in_leaf, random_state=123)
    lgbm.fit(X, y)
    return lgbm


def tune_rf(X, y):
    n_estimators = 50  # [50, 100, 200]
    max_depth = None  # [None, 10, 20]
    min_samples_split = 2  # [2, 5, 10]
    min_samples_leaf = 1  # [1, 2, 4]
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=123)
    rf.fit(X, y)
    return rf


def tune_svm(X, y):
    C = 10  # [0.1, 1, 10, 100]
    gamma = 0.01  # [1, 0.1, 0.01, 0.001]
    svm = SVR(C=C, gamma=gamma, kernel='rbf')
    svm.fit(X, y)
    return svm


def tune_nn(X, y):
    input_shape = (X.shape[1],)  # Infer input shape from X

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    epochs = 100  # [10, 50, 100, 200]
    batch_size = 10  # [10, 20, 30]
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return model


def tune_gbr(X, y, n):
    n_estimators = 100
    max_depth = 6
    min_samples_leaf = n
    gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    gbr.fit(X, y)
    return gbr
