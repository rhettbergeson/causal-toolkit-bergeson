# Wrote the code myself, and used Claude Opus 5.5 to verify correctness
import pandas as pd
import numpy as np
from typing import Tuple
import math
import scipy.stats as stats


# Calculate the confidence interval for the Average Treatment Effect.
def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    # Split into treated/untreated
    treated_df = data[data['T'] == 1]
    untreated_df = data[data['T'] == 0]
    # Calculate ATE
    ATE_estimate = float(treated_df['Y'].mean() - untreated_df['Y'].mean())
    # Calculate Standard Error
    se = math.sqrt(treated_df['Y'].sem() ** 2 + untreated_df['Y'].sem() ** 2)
    # Build confidence interval
    ci = stats.norm.interval((1-alpha), ATE_estimate, se)
    CI_lower = float(ci[0])
    CI_upper = float(ci[1])
    # Results
    results = (ATE_estimate, CI_lower, CI_upper)
    return results
    


# Calculate the p-value for testing whether the ATE differs from a null hypothesis value.
def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    # Split into treated/untreated
    treated_df = data[data['T'] == 1]
    untreated_df = data[data['T'] == 0]
    # Calculate ATE
    ATE_estimate = float(treated_df['Y'].mean() - untreated_df['Y'].mean())
    # Calculate Standard Error
    se = math.sqrt(treated_df['Y'].sem() ** 2 + untreated_df['Y'].sem() ** 2)
    # Calculate Test Stat & P-Value
    t_statistic = (ATE_estimate - 0) / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_statistic))))
    # Results
    results = (ATE_estimate, t_statistic, p_value)
    return results