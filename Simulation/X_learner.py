import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from Simulation.Tuning import tune_lgbm, tune_rf, tune_svm, tune_nn, tune_gbr


def X_learner_lgbm(train, test, X, T, y):
    m0 = tune_lgbm(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_lgbm(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])

    # propensity score model
    g = LogisticRegression(solver="lbfgs", penalty='none', random_state=123)
    g.fit(train[X], train[T])

    # Estimate treatment effects
    d_train = np.where(train[T] == 0,
                       m1.predict(train[X]) - train[y],
                       train[y] - m0.predict(train[X]))

    # second stage models
    mx0 = tune_lgbm(train.query(f"{T}==0")[X], d_train[train[T] == 0])
    mx1 = tune_lgbm(train.query(f"{T}==1")[X], d_train[train[T] == 1])

    def ps_predict(df, t):
        return g.predict_proba(df[X])[:, t]

    x_cate_train = train.assign(pred_cate=(ps_predict(train, 1) * mx0.predict(train[X]) +
                                           ps_predict(train, 0) * mx1.predict(train[X])))

    x_cate_test = test.assign(pred_cate=(ps_predict(test, 1) * mx0.predict(test[X]) +
                                         ps_predict(test, 0) * mx1.predict(test[X])))

    return x_cate_train, x_cate_test


def X_learner_linear(train, test, X, T, y):
    m0 = LinearRegression()
    m1 = LinearRegression()

    # propensity score model
    g = LogisticRegression(solver="lbfgs", penalty='none', random_state=123)
    g.fit(train[X], train[T])

    m0.fit(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1.fit(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])

    # Estimate treatment effects
    d_train = np.where(train[T] == 0,
                       m1.predict(train[X]) - train[y],
                       train[y] - m0.predict(train[X]))

    # second stage
    mx0 = LinearRegression()
    mx1 = LinearRegression()
    mx0.fit(train.query(f"{T}==0")[X], d_train[train[T] == 0])
    mx1.fit(train.query(f"{T}==1")[X], d_train[train[T] == 1])

    def ps_predict(df, t):
        return g.predict_proba(df[X])[:, t]

    x_cate_train = train.assign(pred_cate=(ps_predict(train, 1) * mx0.predict(train[X]) +
                                           ps_predict(train, 0) * mx1.predict(train[X])))

    x_cate_test = test.assign(pred_cate=(ps_predict(test, 1) * mx0.predict(test[X]) +
                                         ps_predict(test, 0) * mx1.predict(test[X])))

    return x_cate_train, x_cate_test


def X_learner_rf(train, test, X, T, y):
    m0 = tune_rf(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_rf(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])

    # propensity score model
    g = LogisticRegression(solver="lbfgs", penalty='none', random_state=123)
    g.fit(train[X], train[T])

    d_train = np.where(train[T] == 0,
                       m1.predict(train[X]) - train[y],
                       train[y] - m0.predict(train[X]))

    # second stage
    mx0 = tune_rf(train.query(f"{T}==0")[X], d_train[train[T] == 0])
    mx1 = tune_rf(train.query(f"{T}==1")[X], d_train[train[T] == 1])

    def ps_predict(df, t):
        return g.predict_proba(df[X])[:, t]

    x_cate_train = train.assign(pred_cate=(ps_predict(train, 1) * mx0.predict(train[X]) +
                                           ps_predict(train, 0) * mx1.predict(train[X])))

    x_cate_test = test.assign(pred_cate=(ps_predict(test, 1) * mx0.predict(test[X]) +
                                         ps_predict(test, 0) * mx1.predict(test[X])))

    return x_cate_train, x_cate_test


def X_learner_svm(train, test, X, T, y):
    m0 = tune_svm(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_svm(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])

    # propensity score model
    g = LogisticRegression(solver="lbfgs", penalty='none', random_state=123)
    g.fit(train[X], train[T])

    d_train = np.where(train[T] == 0,
                       m1.predict(train[X]) - train[y],
                       train[y] - m0.predict(train[X]))

    # second stage
    mx0 = tune_svm(train.query(f"{T}==0")[X], d_train[train[T] == 0])
    mx1 = tune_svm(train.query(f"{T}==1")[X], d_train[train[T] == 1])

    def ps_predict(df, t):
        return g.predict_proba(df[X])[:, t]

    x_cate_train = train.assign(pred_cate=(ps_predict(train, 1) * mx0.predict(train[X]) +
                                           ps_predict(train, 0) * mx1.predict(train[X])))

    x_cate_test = test.assign(pred_cate=(ps_predict(test, 1) * mx0.predict(test[X]) +
                                         ps_predict(test, 0) * mx1.predict(test[X])))

    return x_cate_train, x_cate_test