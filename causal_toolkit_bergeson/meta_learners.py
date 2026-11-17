# Wrote most of the code myself, and used Claude Opus 5.5 to correct mistakes & test
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np



# S-Learner
def s_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    # Fit the S-Learner
    s_learner = LGBMRegressor()
    s_learner.fit(train[X+[T]], train[y])
    # Filter Test dataframe
    test_filtered = test[X + [T]]
    # Create treated/control dataframes
    test_0 = test_filtered.assign(**{T: 0})
    test_1 = test_filtered.assign(**{T: 1})
    # Define CATE
    cate = s_learner.predict(test_1) - s_learner.predict(test_0)
    # Apply CATE
    s_learner_cate_test = test.assign(cate=cate)
    # Return the dataframe
    return s_learner_cate_test


# T-Learner
def t_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    # Create models
    m0 = LGBMRegressor()
    m1 = LGBMRegressor()
    # Fit models
    m0.fit(train.query(f"{T}==0")[X], train.query(f"{T}==0")[y])
    m1.fit(train.query(f"{T}==1")[X], train.query(f"{T}==1")[y])
    # Assign values
    t_learner_cate_test = test.assign(
    cate=m1.predict(test[X]) - m0.predict(test[X]))
    # return Dataframe
    return t_learner_cate_test



# X-Learner
def x_learner_discrete(train, test, X, T, y) -> pd.DataFrame:
    # Split train by treatment group
    ctrl = train[train[T] == 0]
    trt  = train[train[T] == 1]
    # Outcome models (same as T-learner)
    m0 = LGBMRegressor().fit(ctrl[X], ctrl[y])
    m1 = LGBMRegressor().fit(trt[X], trt[y])
    # Imputed treatment effects
    tau_0 = m1.predict(ctrl[X]) - ctrl[y]   # controls: imputed treated outcome minus actual
    tau_1 = trt[y] - m0.predict(trt[X])     # treated: actual minus imputed control outcome
    # Propensity score e(x) on test
    ps_model = LogisticRegression(penalty=None).fit(train[X], train[T])
    ps = ps_model.predict_proba(test[X])[:, 1]   # column 1 = P(T=1 | x)
    # X Learner models
    mu_tau0 = LGBMRegressor().fit(ctrl[X], tau_0)
    mu_tau1 = LGBMRegressor().fit(trt[X], tau_1)
    # Calculate CATE: propensity-weighted blend of the two effect models
    cate = ((1 - ps) * mu_tau1.predict(test[X]) +
        ps * mu_tau0.predict(test[X]))
    # Return Dataframe
    return test.assign(cate=cate)



# Double ML
def double_ml_cate(train, test, X, T, y) -> pd.DataFrame:
    # Models
    debias_m = LGBMRegressor()
    denoise_m = LGBMRegressor()
    # Residuals
    t_res =  train[T] - cross_val_predict(debias_m,train[X],train[T],cv=5)
    y_res =  train[y] - cross_val_predict(denoise_m,train[X],train[y],cv=5)
    # Transformed outcomes and weights
    y_star = y_res/t_res
    w = t_res**2
    # Build CATE model
    cate_model = LGBMRegressor().fit(train[X], y_star, sample_weight=w)
    # Predict CATE on test & return Dataframe
    return test.assign(cate=cate_model.predict(test[X]))
