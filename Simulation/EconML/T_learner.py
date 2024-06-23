from econml.metalearners import SLearner, TLearner
from sklearn.ensemble import GradientBoostingRegressor


def T_learner(train, test, X, T, y):
    n = len(train[y])
    models = GradientBoostingRegressor(n_estimators=100, max_depth=20, min_samples_leaf=int(n/100),
                                              random_state=123)
    t = TLearner(models=models)
    t.fit(train[y], train[T], X=train[X])

    t_learner_cate_train = train.assign(
        pred_cate=(t.effect(train[X]))
    )

    t_learner_cate_test = test.assign(
        pred_cate=(t.effect(test[X]))
    )

    return t_learner_cate_train, t_learner_cate_test