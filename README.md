# causal-toolkit-bergeson

[![Tests](https://github.com/rhettbergeson/causal-toolkit-bergeson/actions/workflows/tests.yml/badge.svg)](https://github.com/rhettbergeson/causal-toolkit-bergeson/actions)
[![PyPI version](https://badge.fury.io/py/causal-toolkit-bergeson.svg)](https://pypi.org/project/causal-toolkit-bergeson/)

A Python package for estimating causal effects. It collects the methods built in
UChicago's *Causal Models in Data Science* (ADSP 32029) course into one installable
toolkit:

- **Randomized experiments**: average treatment effect (ATE) with confidence
  intervals and p-values
- **Propensity score methods**: inverse propensity weighting (IPW) and doubly
  robust (AIPW) estimation of the ATE
- **Meta-learners**: S-, T- and X-learners and Double ML for estimating
  conditional average treatment effects (CATE)

## Installation

Requires Python 3.8+. Install from [PyPI](https://pypi.org/project/causal-toolkit-bergeson/):

```bash
pip install causal-toolkit-bergeson
```

### From source (for development)

Clone the repository and install in editable mode with [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/rhettbergeson/causal-toolkit-bergeson.git
cd causal-toolkit-bergeson

# Install in editable mode
uv pip install -e .

# Run the test suite to verify the installation
uv run pytest
```

The package installs as `causal-toolkit-bergeson` but is imported as
`causal_toolkit_bergeson` (Python import names can't contain hyphens).

Using plain pip instead? `pip install -e ".[dev]"` installs the package along with
the testing and linting tools.

## Quick start

```python
import numpy as np
import pandas as pd
from causal_toolkit_bergeson import (
    calculate_ate_ci, calculate_ate_pvalue, ipw, doubly_robust,
    s_learner_discrete, t_learner_discrete, x_learner_discrete, double_ml_cate,
)

rng = np.random.default_rng(0)
n = 2000
```

### Randomized experiment

The RCT functions expect a binary treatment column named `T` and an outcome
column named `Y`.

```python
rct = pd.DataFrame({"T": rng.binomial(1, 0.5, n)})
rct["Y"] = 1.5 * rct["T"] + rng.normal(0, 2, n)      # true ATE = 1.5

ate, ci_lower, ci_upper = calculate_ate_ci(rct, alpha=0.05)
ate, t_stat, p_value = calculate_ate_pvalue(rct)
# ATE ≈ 1.39, 95% CI ≈ (1.21, 1.56), p < 0.001
```

### Observational data (confounded treatment)

```python
x1, x2 = rng.normal(size=n), rng.normal(size=n)
t = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2))))  # x1, x2 drive treatment
y = 2.0 * t + x1 + 0.5 * x2 + rng.normal(0, 0.5, n)            # true ATE = 2.0
obs = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

# ATE with propensity score methods
ipw(obs, "x1 + x2", T="t", Y="y")             # ≈ 1.99
doubly_robust(obs, "x1 + x2", T="t", Y="y")   # ≈ 1.98

# CATE with meta-learners: fit on train, score test
train, test = obs.iloc[:1600], obs.iloc[1600:]
X = ["x1", "x2"]

s_out  = s_learner_discrete(train, test, X, "t", "y")
t_out  = t_learner_discrete(train, test, X, "t", "y")
x_out  = x_learner_discrete(train, test, X, "t", "y")
dml_out = double_ml_cate(train, test, X, "t", "y")

dml_out["cate"].mean()   # ≈ 2.00 (S/T/X-learners give ≈ 2.02–2.04)
```

## API reference

All functions can be imported from the top level of the package.

### Randomized experiments (`rct.py`)

#### `calculate_ate_ci(data, alpha=0.05)`

Difference-in-means ATE with a normal-approximation confidence interval.

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame` | Must contain a binary treatment column `T` (0/1) and an outcome column `Y` |
| `alpha` | `float` | Significance level; the interval has coverage `1 - alpha` (default `0.05`) |

**Returns** `(ate, ci_lower, ci_upper)`, a tuple of floats. The standard error is
`sqrt(SE_treated² + SE_control²)`.

#### `calculate_ate_pvalue(data)`

Two-sided test of H₀: ATE = 0, using a normal approximation.

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame` | Must contain columns `T` (0/1) and `Y` |

**Returns** `(ate, t_statistic, p_value)`, a tuple of floats.

### Propensity score methods (`propensity.py`)

Both functions estimate the propensity score e(x) = P(T = 1 | X) with an
unpenalized logistic regression. The covariates are specified as a
[patsy](https://patsy.readthedocs.io/) formula, for example `"x1 + x2"` or
`"age + C(region)"`.

#### `ipw(df, ps_formula, T, Y)`

ATE by inverse propensity weighting: mean of `(T − e(x)) / (e(x)(1 − e(x))) · Y`.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Data containing the treatment, outcome and covariates |
| `ps_formula` | `str` | Patsy right-hand-side formula for the propensity model |
| `T` | `str` | Name of the binary treatment column |
| `Y` | `str` | Name of the outcome column |

**Returns** the estimated ATE as a `float`.

#### `doubly_robust(df, formula, T, Y)`

ATE by augmented IPW. The estimator combines the propensity model with separate
linear outcome models for the treated and control groups. It stays consistent
if *either* the propensity model or the outcome models are correctly specified.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Data containing the treatment, outcome and covariates |
| `formula` | `str` | Patsy formula used for both the propensity and outcome models |
| `T` | `str` | Name of the binary treatment column |
| `Y` | `str` | Name of the outcome column |

**Returns** the estimated ATE as a `float`.

### Meta-learners (`meta_learners.py`)

All four functions share the same signature and return value. They fit models
on `train` and estimate the CATE for every row of `test`. The underlying
models are LightGBM regressors with default hyperparameters.

| Parameter | Type | Description |
|---|---|---|
| `train` | `pd.DataFrame` | Training data (covariates, treatment and outcome) |
| `test` | `pd.DataFrame` | Data to score; it needs the covariate columns |
| `X` | `list[str]` | Covariate column names |
| `T` | `str` | Treatment column name |
| `y` | `str` | Outcome column name |

**Returns** a copy of `test` with an added `cate` column.

| Function | Treatment | Method |
|---|---|---|
| `s_learner_discrete` | Binary | One model on `X + T`. CATE = prediction with T=1 minus prediction with T=0 |
| `t_learner_discrete` | Binary | Separate outcome models for the treated and control groups. CATE = difference of their predictions |
| `x_learner_discrete` | Binary | Imputes individual effects from T-learner models, models them within each group, then blends with a logistic propensity score |
| `double_ml_cate` | Binary **or continuous** | Residualizes T and Y on X with 5-fold cross-fitting, then fits a weighted model to the ratio `Ỹ / T̃` |

## Running the tests

```bash
uv run pytest
```

This runs the suite in `tests/` with a coverage report: terminal output plus an
HTML report in `htmlcov/`.

## Notes

- LightGBM prints `[LightGBM] [Info]` messages while it trains. These are
  informational, not errors.

## License

MIT. See [LICENSE](https://github.com/rhettbergeson/causal-toolkit-bergeson/blob/main/LICENSE).

---

*AI disclosure: README drafted with Claude Opus 5.5 (Anthropic) and reviewed by me.*