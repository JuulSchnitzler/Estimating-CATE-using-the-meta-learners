# Variables to store selected features
outcome_selection = ["age", "weight", "po2", "pco2", "ph", "lung_compliance", "hco3", "minute_volume",
                     "respiratory_rate"]
L = ["age", "weight", "pf_ratio", "po2", "driving_pressure", "fio2", "hco3", "plateau_pressure", "respiratory_rate"]
T = "peep_regime"
X_rct = ["age", "sex", "weight", "height", "pf_ratio", "po2", "pco2", "ph", "driving_pressure", "lung_compliance",
         "fio2", "hco3", "heart_rate", "minute_volume", "peep", "plateau_pressure", "respiratory_rate",
         "syst_blood_pressure"]


# Takes as input the RCT dataset, impute object, and normalizer object.
# Returns preprocessed dataset
def preprocessing(RCT_data, impute, normalizer):
    # Additional preprocessing before imputation
    RCT_data['sex'].replace(['F', 'M'], [0, 1], inplace=True)
    RCT_data['peep_regime'].replace(['low', 'high'], [0, 1], inplace=True)
    RCT_data['mort_28'].replace([False, True], [0, 1], inplace=True)

    RCT_data[X_rct] = impute.transform(RCT_data[X_rct])
    RCT_data[X_rct] = normalizer.transform(RCT_data[X_rct])

    return RCT_data


def X_learner(g, mx0, mx1, RCT_data):
    def ps_predict(df, t):
        return g.predict_proba(df[L])[:, t]

    pred_cate = (ps_predict(RCT_data, 1) * mx0.predict(RCT_data[outcome_selection]) +
                 ps_predict(RCT_data, 0) * mx1.predict(RCT_data[outcome_selection]))

    return pred_cate


def T_learner(m0, m1, RCT_data):
    # CATE estimation
    pred_cate = m1.predict(RCT_data[L]) - m0.predict(RCT_data[L])
    return pred_cate


def S_learner(m, RCT_data):
    # CATE estimation
    pred_cate = m.predict(RCT_data[L].assign(**{T: 1})) - m.predict(RCT_data[L].assign(**{T: 0}))
    return pred_cate
