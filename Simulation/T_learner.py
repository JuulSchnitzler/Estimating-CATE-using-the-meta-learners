from sklearn.linear_model import LinearRegression
from Simulation.Tuning import tune_lgbm, tune_rf, tune_svm, tune_nn


def T_learner(m0, m1, train, test, X, T, y):
    # estimate the CATE
    t_learner_cate_train = train.assign(pred_cate=m1.predict(train[X]) - m0.predict(train[X]))
    t_learner_cate_test = test.assign(pred_cate=m1.predict(test[X]) - m0.predict(test[X]))
    return t_learner_cate_train, t_learner_cate_test


def T_learner_lgbm(train, test, X, T, y):
    m0 = tune_lgbm(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_lgbm(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    return T_learner(m0, m1, train, test, X, T, y)


def T_learner_linear(train, test, X, T, y):
    m0 = LinearRegression()
    m1 = LinearRegression()
    m0.fit(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1.fit(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    return T_learner(m0, m1, train, test, X, T, y)


def T_learner_rf(train, test, X, T, y):
    m0 = tune_rf(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_rf(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    return T_learner(m0, m1, train, test, X, T, y)


def T_learner_svm(train, test, X, T, y):
    m0 = tune_svm(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_svm(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    return T_learner(m0, m1, train, test, X, T, y)


def T_learner_nn(train, test, X, T, y):
    m0 = tune_nn(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1 = tune_nn(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    return T_learner(m0, m1, train, test, X, T, y)

