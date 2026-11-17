# Wrote the code myself, and used Claude Opus 5.5 to verify correctness
import pandas as pd
import numpy as np
import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression




# Calculate the Average Treatment Effect using Inverse Propensity Weighting.
def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    X = patsy.dmatrix(ps_formula, df)
    ps_model = LogisticRegression(penalty=None, max_iter=1000).fit(X, df[T])
    ps = ps_model.predict_proba(X)[:, 1]
    # Compute ATE
    return np.mean((df[T]-ps) / (ps*(1-ps)) * df[Y])





# Calculate the Average Treatment Effect using Doubly Robust Estimation.
def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    X = patsy.dmatrix(formula, df)
    ps_model = LogisticRegression(penalty=None, max_iter=1000).fit(X, df[T])
    ps = ps_model.predict_proba(X)[:, 1]
    # Fit Outcome Models
    m0 = LinearRegression().fit(X[df[T]==0, :], df.query(f"{T}==0")[Y])
    m1 = LinearRegression().fit(X[df[T]==1, :], df.query(f"{T}==1")[Y])
    # Get predictions
    m0_hat = m0.predict(X)
    m1_hat = m1.predict(X)
    # Calculate ATE using doubly robust formula
    return (
        np.mean(df[T]*(df[Y] - m1_hat)/ps + m1_hat) -
        np.mean((1-df[T])*(df[Y] - m0_hat)/(1-ps) + m0_hat)
    )

