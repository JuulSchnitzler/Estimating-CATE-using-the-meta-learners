from sklearn.model_selection import train_test_split
from Simulation.Simulation_setup import generate_data, make_dataframe, MSE

from Simulation.S_learner import S_learner_linear, S_learner_lgbm, S_learner_rf, S_learner_svm, S_learner_nn
from Simulation.T_learner import T_learner_linear, T_learner_rf, T_learner_lgbm, T_learner_svm, T_learner_nn
from Simulation.X_learner import X_learner_linear, X_learner_rf, X_learner_lgbm, X_learner_svm, X_learner_nn


# Perform multiple experiments with different sample size
def perform_experiments(N, e, d, mu0, mu1, model):
    s_mse_list = []
    t_mse_list = []
    x_mse_list = []

    for n in N:
        print('Num samples:', n)
        # Generate data for n number of samples
        X, T, Y0, Y1, Y = generate_data(n, e, d, mu0, mu1)
        df_sim, X_sim = make_dataframe(X, T, Y0, Y1, Y)
        train_sim, test_sim = train_test_split(df_sim, test_size=0.3, random_state=13)

        if model == "LGBM":
            # Get CATE estimates (for S-, T- and X-learner)
            s_cate_train, s_cate_test = S_learner_lgbm(train_sim, test_sim, X_sim, 'T', 'Y')
            t_cate_train, t_cate_test = T_learner_lgbm(train_sim, test_sim, X_sim, 'T', 'Y')
            x_cate_train, x_cate_test = X_learner_lgbm(train_sim, test_sim, X_sim, 'T', 'Y')

        elif model == "LinearRegression":
            # Get CATE estimates (for S-, T- and X-learner)
            s_cate_train, s_cate_test = S_learner_linear(train_sim, test_sim, X_sim, 'T', 'Y')
            t_cate_train, t_cate_test = T_learner_linear(train_sim, test_sim, X_sim, 'T', 'Y')
            x_cate_train, x_cate_test = X_learner_linear(train_sim, test_sim, X_sim, 'T', 'Y')

        elif model == "RF":
            # Get CATE estimates (for S-, T- and X-learner)
            s_cate_train, s_cate_test = S_learner_rf(train_sim, test_sim, X_sim, 'T', 'Y')
            t_cate_train, t_cate_test = T_learner_rf(train_sim, test_sim, X_sim, 'T', 'Y')
            x_cate_train, x_cate_test = X_learner_rf(train_sim, test_sim, X_sim, 'T', 'Y')

        elif model == "SVM":
            # Get CATE estimates (for S-, T- and X-learner)
            s_cate_train, s_cate_test = S_learner_svm(train_sim, test_sim, X_sim, 'T', 'Y')
            t_cate_train, t_cate_test = T_learner_svm(train_sim, test_sim, X_sim, 'T', 'Y')
            x_cate_train, x_cate_test = X_learner_svm(train_sim, test_sim, X_sim, 'T', 'Y')

        else:
            s_cate_train, s_cate_test = None, None
            t_cate_train, t_cate_test = None, None
            x_cate_train, x_cate_test = None, None

        # Calculate MSE
        s_mse_test = MSE(s_cate_test)
        t_mse_test = MSE(t_cate_test)
        x_mse_test = MSE(x_cate_test)

        s_mse_list.append(s_mse_test)
        t_mse_list.append(t_mse_test)
        x_mse_list.append(x_mse_test)

    return s_mse_list, t_mse_list, x_mse_list


def iterate_experiments(N, num_experiments, e, d, mu0, mu1, model):
    s_mse_total = []
    t_mse_total = []
    x_mse_total = []

    for i in range(num_experiments):
        print('round', i)
        s_mse_list, t_mse_list, x_mse_list = perform_experiments(N, e, d, mu0, mu1, model)
        s_mse_total.append(s_mse_list)
        t_mse_total.append(t_mse_list)
        x_mse_total.append(x_mse_list)

    return s_mse_total, t_mse_total, x_mse_total
