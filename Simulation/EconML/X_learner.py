from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def X_learner_lgbm(train, test, X, T, y):
    models = LGBMRegressor(max_depth=20, min_child_samples=10, random_state=123)
    propensity_model = LogisticRegression(solver="lbfgs", penalty='none', random_state=123)
    x = XLearner(models=models, propensity_model=propensity_model)
    x.fit(train[y], train[T], X=train[X])

    x_learner_cate_train = train.assign(
        pred_cate=(x.effect(train[X]))
    )

    x_learner_cate_test = test.assign(
        pred_cate=(x.effect(test[X]))
    )

    return x_learner_cate_train, x_learner_cate_test


def X_learner_gbr(train, test, X, T, y):
    n = len(train[y])
    models = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n / 100))
    propensity_model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n / 100))
    x = XLearner(models=models, propensity_model=propensity_model)
    x.fit(train[y], train[T], X=train[X])

    x_learner_cate_train = train.assign(
        pred_cate=(x.effect(train[X]))
    )

    x_learner_cate_test = test.assign(
        pred_cate=(x.effect(test[X]))
    )

    return x_learner_cate_train, x_learner_cate_test