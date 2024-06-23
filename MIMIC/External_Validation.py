import os
import pickle
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Load dataset
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, "impute_mimic.csv")
mimic_complete = pd.read_csv(file_path)

# Create variables to store outcome Y, treatment T, and features X
y = "mort_28"
T = "peep_regime"
L = ["age", "weight", "pf_ratio", "po2", "driving_pressure", "fio2", "hco3", "plateau_pressure", "respiratory_rate"]
X_rct = ["age", "sex", "weight", "height", "pf_ratio", "po2", "pco2", "ph", "driving_pressure", "lung_compliance",
         "fio2", "hco3", "heart_rate", "minute_volume", "peep", "plateau_pressure", "respiratory_rate",
         "syst_blood_pressure"]
feature_selection = ["age", "weight", "po2", "pco2", "ph", "lung_compliance", "hco3", "minute_volume",
                     "respiratory_rate"]

# Save imputation model
imputeObject = IterativeImputer(random_state=0, max_iter=20)
imputeObject.fit(mimic_complete[X_rct])
pickle.dump(imputeObject, open('../External Validation/impute-model.pk1', 'wb'))

# Save normalization model
normalizer = MinMaxScaler()
normalizer.fit(mimic_complete[X_rct])
mimic_complete[X_rct] = normalizer.transform(mimic_complete[X_rct])
pickle.dump(normalizer, open('../External Validation/normalizer-model.pk1', 'wb'))

# X-learner using SVR
# First stage models
m0 = SVR(C=10, gamma=0.01, kernel='rbf')
m1 = SVR(C=10, gamma=0.01, kernel='rbf')
m0.fit(mimic_complete.query(f"{T}==0")[L], mimic_complete.query(f"{T}==0")[y])
m1.fit(mimic_complete.query(f"{T}==1")[L], mimic_complete.query(f"{T}==1")[y])

# propensity score model
g = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0, random_state=123)
g.fit(mimic_complete[L], mimic_complete[T])

# Estimate treatment effects
d_train = np.where(mimic_complete[T] == 0,
                   m1.predict(mimic_complete[L]) - mimic_complete[y],
                   mimic_complete[y] - m0.predict(mimic_complete[L]))

# Second stage models
mx0 = SVR(C=10, gamma=0.01, kernel='rbf')
mx1 = SVR(C=10, gamma=0.01, kernel='rbf')
mx0.fit(mimic_complete.query(f"{T}==0")[feature_selection], d_train[mimic_complete[T] == 0])
mx1.fit(mimic_complete.query(f"{T}==1")[feature_selection], d_train[mimic_complete[T] == 1])

# Save models X-learner
pickle.dump(g, open('../External Validation/X-learner/x-learner-model-g.pk1', 'wb'))
pickle.dump(mx0, open('../External Validation/X-learner/x-learner-model-mu0.pk1', 'wb'))
pickle.dump(mx1, open('../External Validation/X-learner/x-learner-model-mu1.pk1', 'wb'))

# Save models T-learner
m0 = SVR(C=10, gamma=0.01, kernel='rbf')
m1 = SVR(C=10, gamma=0.01, kernel='rbf')
m0.fit(mimic_complete.query(f"{T}==0")[L], mimic_complete.query(f"{T}==0")[y])
m1.fit(mimic_complete.query(f"{T}==1")[L], mimic_complete.query(f"{T}==1")[y])
pickle.dump(m0, open('../External Validation/T-learner/t-learner-model-m0.pk1', 'wb'))
pickle.dump(m1, open('../External Validation/T-learner/t-learner-model-m1.pk1', 'wb'))

# Save model S-learner
s_learner = SVR(C=10, gamma=0.01, kernel='rbf')
s_learner.fit(mimic_complete[L + [T]], mimic_complete[y])
pickle.dump(s_learner, open('../External Validation/S-learner/s-learner-model.pk1', 'wb'))
