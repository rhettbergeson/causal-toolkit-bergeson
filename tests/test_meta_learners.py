# AI disclosure: Test code drafted with Claude Opus 5.5 (Anthropic); reviewed and run by me.
#
# Pytest tests for the meta-learners in causal_toolkit_bergeson/meta_learners.py.
# Run from the repository root with:  uv run pytest
#
# How pytest works here (quick primer):
#   - pytest finds every function whose name starts with `test_` and runs it.
#   - A test PASSES if it finishes without raising an exception.
#   - A test FAILS if any `assert` is False (or the code under test crashes).
#   - simple_data() / continuous_treatment_data() do NOT start with `test_`,
#     so pytest treats them as ordinary helpers, not tests.

import numpy as np
import pandas as pd

# Imported from the package's top level so __init__.py exports are exercised too
from causal_toolkit_bergeson import (
    s_learner_discrete,
    t_learner_discrete,
    x_learner_discrete,
    double_ml_cate,
)


# =============================================================================
# Test data generation (provided by the assignment; copied as-is)
# NOTE: Not decorated with @pytest.fixture because Gradescope doesn't support
# it here. Each test calls these directly, so data is regenerated per test.
# The fixed seeds make every call return identical data, so tests are
# deterministic (same result every run).
# =============================================================================

def simple_data():
    """Generate simple data with known treatment effect"""
    np.random.seed(42)
    n = 1000

    # Covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Treatment assignment (confounded)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)

    # Outcome with constant treatment effect = 2.0
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

    # Split into train/test
    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


def continuous_treatment_data():
    """Generate data with continuous treatment"""
    np.random.seed(789)
    n = 1000

    # Covariates
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Continuous treatment
    t = 10 + x1 + 2*x2 + np.random.normal(0, 1, n)

    # Outcome: linear effect of treatment
    y = t + x1 + 0.5*x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 't': t, 'y': y})

    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


# =============================================================================
# Shared constants
# Column names match the data generators above.
# TRUE_EFFECT comes from the data-generating process: y = 2.0 * t + ...
# TOLERANCE is deliberately loose (25% of the true effect). LightGBM with
# default settings on 800 rows won't hit 2.0 exactly, and S-learners in
# particular tend to shrink the effect toward 0 (the tree may not split on t
# everywhere). A tight tolerance would make the test "flaky" — failing for
# reasons of estimation noise rather than an actual bug.
# =============================================================================

X = ['x1', 'x2']
T = 't'
Y = 'y'

TRUE_EFFECT = 2.0            # binary-treatment data: constant CATE of 2.0
TRUE_EFFECT_CONTINUOUS = 1.0 # continuous-treatment data: dy/dt = 1.0
TOLERANCE = 0.5


# =============================================================================
# S-Learner tests (5)
# Pattern for every test: 1) generate data  2) call function  3) assert
# =============================================================================

def test_s_learner_returns_dataframe():
    """S-learner output should be a pandas DataFrame."""
    train, test = simple_data()

    result = s_learner_discrete(train, test, X, T, Y)

    assert isinstance(result, pd.DataFrame)


def test_s_learner_has_cate_column():
    """S-learner output should contain a 'cate' column, with one row per test row."""
    train, test = simple_data()

    result = s_learner_discrete(train, test, X, T, Y)

    assert 'cate' in result.columns
    # Extra guard: the function should score every test row (no rows dropped/added)
    assert len(result) == len(test)


def test_s_learner_constant_effect():
    """With a constant true effect of 2.0, the average estimated CATE should be close to 2.0."""
    train, test = simple_data()

    result = s_learner_discrete(train, test, X, T, Y)

    # The ATE is the mean of the CATEs across the test set. We compare the
    # ATE (not each individual CATE) because individual tree predictions are
    # noisy; averaging over 200 rows smooths that noise out.
    estimated_ate = result['cate'].mean()
    assert abs(estimated_ate - TRUE_EFFECT) < TOLERANCE, (
        f"Estimated ATE {estimated_ate:.3f} not within {TOLERANCE} of {TRUE_EFFECT}"
    )


def test_s_learner_return_numeric_cate():
    """CATE values should be a numeric dtype (e.g., float), not object/string."""
    train, test = simple_data()

    result = s_learner_discrete(train, test, X, T, Y)

    assert pd.api.types.is_numeric_dtype(result['cate'])


def test_s_learner_no_nan_values():
    """No CATE value should be NaN."""
    train, test = simple_data()

    result = s_learner_discrete(train, test, X, T, Y)

    # .isna().any() is True if at least one NaN exists, so we assert it's False
    assert not result['cate'].isna().any()


# =============================================================================
# T-Learner and X-Learner tests (1 each)
# =============================================================================

def test_t_learner_returns_dataframe():
    """T-learner output should be a pandas DataFrame."""
    train, test = simple_data()

    result = t_learner_discrete(train, test, X, T, Y)

    assert isinstance(result, pd.DataFrame)


def test_x_learner_returns_dataframe():
    """X-learner output should be a pandas DataFrame."""
    train, test = simple_data()

    result = x_learner_discrete(train, test, X, T, Y)

    assert isinstance(result, pd.DataFrame)


# =============================================================================
# Double ML tests (2)
# =============================================================================

def test_double_ml_returns_dataframe():
    """Double ML output should be a pandas DataFrame (binary-treatment data)."""
    train, test = simple_data()

    result = double_ml_cate(train, test, X, T, Y)

    assert isinstance(result, pd.DataFrame)


def test_double_ml_continuous_treatment():
    """Double ML should run on a continuous treatment and return a usable result.

    Why only Double ML gets this test: the S/T/X-learners above are the
    "_discrete" versions — they split or set T to 0/1, which makes no sense for
    a continuous T. Double ML works on residuals (T - E[T|X]), so it handles
    any numeric treatment.
    """
    train, test = continuous_treatment_data()

    result = double_ml_cate(train, test, X, T, Y)

    # "Works with continuous T" = runs without crashing AND returns a usable result
    assert isinstance(result, pd.DataFrame)
    assert 'cate' in result.columns
    assert len(result) == len(test)
    assert pd.api.types.is_numeric_dtype(result['cate'])
    assert not result['cate'].isna().any()