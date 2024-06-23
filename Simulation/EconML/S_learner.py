from econml.metalearners import SLearner
from sklearn.ensemble import GradientBoostingRegressor


def S_learner(train, test, X, T, y):
    n = len(train[y])
    overall_model = GradientBoostingRegressor(n_estimators=100, max_depth=20, min_samples_leaf=int(n/100),
                                              random_state=123)
    s = SLearner(overall_model=overall_model)
    s.fit(train[y], train[T], X=train[X])

    s_learner_cate_train = train.assign(
        pred_cate=(s.effect(train[X]))
    )

    s_learner_cate_test = test.assign(
        pred_cate=(s.effect(test[X]))
    )

    return s_learner_cate_train, s_learner_cate_test
