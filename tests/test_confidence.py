"""Unit tests for the CV-calibrated confidence variants (ML_Reg.calibrate_confidence_params / apply_confidences).

Synthetic data only. Checks the 4 comparison columns are produced, bounded in [0, 1], monotone-decreasing
with the tree-variance std, and that a degenerate (constant real_y) calibration does not crash.
"""
import numpy as np
import pandas as pd

import python.ADME_build_ML  # noqa: F401  (adds ~/Scripts to sys.path so `import ML_Reg` resolves)
import ML_Reg

CONF_COLS = ['conf_labelstd', 'conf_rmse', 'conf_recal', 'conf_conformal']


def _synthetic_cv(n=400, under_disperse=3.0, seed=0):
    """A CV pred_df whose residual scales with the tree-std (so calibration is learnable + under-dispersed)."""
    rng = np.random.default_rng(seed)
    uq = np.abs(rng.normal(0.3, 0.1, n))
    real = rng.normal(0, 1, n)
    pred = real + rng.normal(0, 1, n) * uq * under_disperse
    return pd.DataFrame({'real_y': real, 'pred_y': pred, 'uq_std': uq})


def test_apply_confidences_columns_bounded_and_monotone():
    """All 4 conf_* columns exist, lie in [0,1], and decrease as the tree-std rises."""
    cv = _synthetic_cv()
    calib = ML_Reg.calibrate_confidence_params(cv)
    out = ML_Reg.apply_confidences(cv, calib)
    # all four comparison columns are present
    assert all(c in out.columns for c in CONF_COLS)
    for c in CONF_COLS:
        v = out[c].to_numpy()
        # every confidence is a valid probability-like score in [0, 1]
        assert np.nanmin(v) >= -1e-9 and np.nanmax(v) <= 1 + 1e-9
        # higher tree-variance std -> lower confidence (negative correlation)
        assert np.corrcoef(out['uq_std'], v)[0, 1] < 0


def test_calibration_recovers_underdispersion_slope():
    """The recal fit |resid| ~ a + b*std recovers a positive slope for under-dispersed tree-std."""
    calib = ML_Reg.calibrate_confidence_params(_synthetic_cv(under_disperse=3.0))
    # slope is clearly positive (std tracks error) and rmse_cv is finite
    assert calib['recal_b'] > 0.5
    assert np.isfinite(calib['rmse_cv']) and calib['rmse_cv'] > 0


def test_degenerate_calibration_is_safe():
    """A constant-real_y calibration arm must not crash; columns still produced."""
    deg = pd.DataFrame({'real_y': np.ones(10), 'pred_y': np.linspace(0, 1, 10), 'uq_std': np.linspace(0.1, 1, 10)})
    calib = ML_Reg.calibrate_confidence_params(deg)
    out = ML_Reg.apply_confidences(deg, calib)
    # columns present even though the calibration arm is degenerate
    assert all(c in out.columns for c in CONF_COLS)


def _synthetic_folds(n=500, k=5, seed=0):
    """A CV pred_df with a fold column; residual scales with tree-std (learnable, under-dispersed)."""
    df = _synthetic_cv(n=n, seed=seed).copy()
    df['fold'] = np.random.default_rng(seed).integers(1, k + 1, n)
    df['compound'] = [f'C{i}' for i in range(n)]
    return df


def test_lofo_preserves_order_bounded_monotone():
    """apply_confidences_lofo returns all rows in original order, bounded [0,1], monotone-decreasing with std."""
    pdf = _synthetic_folds()
    out = ML_Reg.apply_confidences_lofo(pdf)
    assert out is not None and len(out) == len(pdf)
    # original row order is restored
    assert (out['compound'].to_numpy() == pdf['compound'].to_numpy()).all()
    for c in CONF_COLS:
        v = out[c].to_numpy()
        assert np.nanmin(v) >= -1e-9 and np.nanmax(v) <= 1 + 1e-9
        assert np.corrcoef(out['uq_std'], v)[0, 1] < 0


def test_lofo_needs_two_folds():
    """LOFO returns None when it can't hold a fold out (single fold, or no fold column)."""
    pdf = _synthetic_folds()
    one = pdf.copy(); one['fold'] = 1
    # a single fold cannot be left out
    assert ML_Reg.apply_confidences_lofo(one) is None
    # missing the fold column entirely
    assert ML_Reg.apply_confidences_lofo(pdf.drop(columns='fold')) is None
