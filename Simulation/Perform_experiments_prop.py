from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from Simulation.Simulation_setup import generate_data, make_dataframe, MSE
from Simulation.X_learner_propensity import X_learner_lgbm_propensity, X_learner_linear_propensity, \
    X_learner_rf_propensity, X_learner_svm_propensity


# Perform multiple experiments with different sample size
def perform_experiments(N, e, d, mu0, mu1, model):
    x_mse_list_logistic = []
    x_mse_list_rf = []
    x_mse_list_dt = []

    for n in N:
        # Generate data for n number of samples
        X, T, Y0, Y1, Y = generate_data(n, e, d, mu0, mu1)
        df_sim, X_sim = make_dataframe(X, T, Y0, Y1, Y)
        train_sim, test_sim = train_test_split(df_sim, test_size=0.3, random_state=13)

        # Different propensity score models
        propensity_logistic = LogisticRegression(solver="lbfgs", C=0.01, random_state=123)
        propensity_rf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_leaf=4, random_state=123)
        propensity_decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=123)

        if model == "LGBM":
            # Get CATE estimates (X-learner)
            x_cate_train_logistic, x_cate_test_logistic = X_learner_lgbm_propensity(train_sim, test_sim, X_sim, 'T',
                                                                                    'Y', propensity_logistic)
            x_cate_train_rf, x_cate_test_rf = X_learner_lgbm_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                        propensity_rf)
            x_cate_train_dt, x_cate_test_dt = X_learner_lgbm_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                        propensity_decision_tree)

        elif model == "LinearRegression":
            # Get CATE estimates (X-learner)
            x_cate_train_logistic, x_cate_test_logistic = X_learner_linear_propensity(train_sim, test_sim, X_sim, 'T',
                                                                                      'Y', propensity_logistic)
            x_cate_train_rf, x_cate_test_rf = X_learner_linear_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                          propensity_rf)
            x_cate_train_dt, x_cate_test_dt = X_learner_linear_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                          propensity_decision_tree)

        elif model == "RF":
            # Get CATE estimates (X-learner)
            x_cate_train_logistic, x_cate_test_logistic = X_learner_rf_propensity(train_sim, test_sim, X_sim, 'T',
                                                                                  'Y', propensity_logistic)
            x_cate_train_rf, x_cate_test_rf = X_learner_rf_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                      propensity_rf)
            x_cate_train_dt, x_cate_test_dt = X_learner_rf_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                      propensity_decision_tree)

        elif model == "SVM":
            # Get CATE estimates (for S-, T- and X-learner)
            x_cate_train_logistic, x_cate_test_logistic = X_learner_svm_propensity(train_sim, test_sim, X_sim, 'T',
                                                                                   'Y', propensity_logistic)
            x_cate_train_rf, x_cate_test_rf = X_learner_svm_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                       propensity_rf)
            x_cate_train_dt, x_cate_test_dt = X_learner_svm_propensity(train_sim, test_sim, X_sim, 'T', 'Y',
                                                                       propensity_decision_tree)

        else:
            x_cate_train_logistic, x_cate_test_logistic = None, None
            x_cate_train_rf, x_cate_test_rf = None, None
            x_cate_train_dt, x_cate_test_dt = None, None

        # Calculate MSE
        x_mse_test_logistic = MSE(x_cate_test_logistic)
        x_mse_test_rf = MSE(x_cate_test_rf)
        x_mse_test_dt = MSE(x_cate_test_dt)

        x_mse_list_logistic.append(x_mse_test_logistic)
        x_mse_list_rf.append(x_mse_test_rf)
        x_mse_list_dt.append(x_mse_test_dt)

    return x_mse_list_logistic, x_mse_list_rf, x_mse_list_dt


def iterate_experiments(N, num_experiments, e, d, mu0, mu1, model):
    x_mse_total_logistic = []
    x_mse_total_rf = []
    x_mse_total_dt = []

    for i in range(num_experiments):
        x_mse_list_logistic, x_mse_list_rf, x_mse_list_dt = perform_experiments(N, e, d, mu0, mu1, model)
        x_mse_total_logistic.append(x_mse_list_logistic)
        x_mse_total_rf.append(x_mse_list_rf)
        x_mse_total_dt.append(x_mse_list_dt)

    return x_mse_total_logistic, x_mse_total_rf, x_mse_total_dt


def hyperparameter_tuning_perform(N, e, d, mu0, mu1):
    # Generate data for N number of samples
    X, T, Y0, Y1, Y = generate_data(N, e, d, mu0, mu1)
    df_sim, X_sim = make_dataframe(X, T, Y0, Y1, Y)
    train_sim, test_sim = train_test_split(df_sim, test_size=0.3, random_state=13)

    # Define hyperparameter grids
    logistic_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [100, 200, 300, 400, 500]}
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 4, 6, 8]
    }
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 4, 6, 8]
    }
    gbc_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 4, 6, 8]
    }

    # Hyperparameter tuning gradient boosting classifier
    gbc_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=123),
                                    gbc_param_grid,
                                    n_iter=25,
                                    random_state=123,
                                    n_jobs=-1,
                                    cv=3,
                                    scoring='accuracy')
    gbc_search.fit(train_sim[X_sim], train_sim['T'])
    propensity_decision_tree = gbc_search.best_estimator_
    best_gbc_params = gbc_search.best_params_

    return 1, 1, 1, best_gbc_params


def hyperparameter_tuning_iterate(N, num_experiments, e, d, mu0, mu1):
    logistic_params_list = []
    rf_params_list = []
    dt_params_list = []
    gbc_params_list = []

    for i in range(num_experiments):
        logistic_params, rf_params, dt_params, gbc_params = hyperparameter_tuning_perform(N, e, d, mu0, mu1)
        logistic_params_list.append(logistic_params)
        rf_params_list.append(rf_params)
        dt_params_list.append(dt_params)
        gbc_params_list.append(gbc_params)

    return logistic_params_list, rf_params_list, dt_params_list, gbc_params_list


def propensity_model_simulation_data(N, e, d, mu0, mu1):
    # Generate data for N number of samples
    X, T, Y0, Y1, Y = generate_data(N, e, d, mu0, mu1)
    df_sim, X_sim = make_dataframe(X, T, Y0, Y1, Y)
    train_sim, test_sim = train_test_split(df_sim, test_size=0.3, random_state=13)

    # Different propensity score models
    propensity_logistic = LogisticRegression(solver="lbfgs", max_iter=100, C=0.01, random_state=123)
    propensity_rf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_leaf=2, random_state=123)
    propensity_decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, random_state=123)

    # Fit your propensity model
    print(train_sim['T'].tolist())

    propensity_logistic.fit(train_sim[X_sim], train_sim['T'])
    propensity_rf.fit(train_sim[X_sim], train_sim['T'])
    propensity_decision_tree.fit(train_sim[X_sim], train_sim['T'])

    # Predict probabilities
    y_probs_logistic = propensity_logistic.predict_proba(test_sim[X_sim])[:, 1]
    y_probs_rf = propensity_rf.predict_proba(test_sim[X_sim])[:, 1]
    y_probs_dt = propensity_decision_tree.predict_proba(test_sim[X_sim])[:, 1]
    y_test = test_sim['T']

    return y_probs_logistic, y_probs_rf, y_probs_dt, y_test
