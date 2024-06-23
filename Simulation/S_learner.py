from sklearn.linear_model import LinearRegression
from Simulation.Tuning import tune_lgbm, tune_rf, tune_svm, tune_nn


def S_learner(s_learner, train, test, X, T, y):
    s_learner_cate_train = train.assign(
        pred_cate=(s_learner.predict(train[X].assign(**{T: 1})) -
                   s_learner.predict(train[X].assign(**{T: 0})))
    )

    s_learner_cate_test = test.assign(
        pred_cate=(s_learner.predict(test[X].assign(**{T: 1})) -  # predict under treatment
                   s_learner.predict(test[X].assign(**{T: 0})))  # predict under control
    )

    return s_learner_cate_train, s_learner_cate_test


def S_learner_lgbm(train, test, X, T, y):
    s_learner = tune_lgbm(train[X + [T]], train[y])
    return S_learner(s_learner, train, test, X, T, y)


def S_learner_linear(train, test, X, T, y):
    s_learner = LinearRegression()
    s_learner.fit(train[X + [T]], train[y])
    return S_learner(s_learner, train, test, X, T, y)


def S_learner_rf(train, test, X, T, y):
    s_learner = tune_rf(train[X + [T]], train[y])
    return S_learner(s_learner, train, test, X, T, y)

def S_learner_svm(train, test, X, T, y):
    s_learner = tune_svm(train[X + [T]], train[y])
    return S_learner(s_learner, train, test, X, T, y)


def S_learner_nn(train, test, X, T, y):
    s_learner = tune_nn(train[X + [T]], train[y])
    return S_learner(s_learner, train, test, X, T, y)
