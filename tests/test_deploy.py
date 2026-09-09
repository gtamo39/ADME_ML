"""Unit tests for OUTPUT.deploy_endpoint (fit internal+public RF, calibrate conf_recal, register to MLTrail).

Synthetic data only (fake feature columns + a fake registry) — no vault write, no real MLTrail, no chemistry.
Checks: the augmented-CV calibration produces the conf_recal params, the model is fit on internal+public,
the bundle carries {model, feature_cols, calibration}, and a NEW H237 model is registered with the recal
scalars mirrored into metrics.
"""
import types
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import python.ADME_build_ML as m   # sets sys.path so `import ML_Reg` resolves; chdir to repo root
from python.ADME_build_ML import OUTPUT


def _fake_params():
    """A PARAMS-like namespace with a small, fast champion RF + one endpoint + a DEPLOY block."""
    return types.SimpleNamespace(
        RF_SINGLETASK={'seed': 42, 'n_jobs': 2,
                       'champion': {'n_estimators': 12, 'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 2}},
        ADME_ENDPOINTS={'sol': {'col': 'sol_Thermodynamic Solubility', 'transform': 'log10', 'unit': 'uM'}},
        DEPLOY={'features_type': 'H237', 'experiment_suffix': '_h237'})


def _fake_data(n_int=60, n_pub=200, n_feat=8, seed=0):
    """A DATA-like namespace: a modelling frame d + int_ids/pub_ids/combo/fold_ids_aug (public in TRAIN only)."""
    rng = np.random.default_rng(seed)
    n = n_int + n_pub
    X = rng.normal(size=(n, n_feat))
    # label correlated with the features (so the RF learns signal and residuals have spread -> learnable calib)
    label = X @ rng.normal(size=n_feat) + rng.normal(scale=0.5, size=n)
    comp = [f'SRB-{i:06d}-001' for i in range(n_int)] + [f'PUB_{i}' for i in range(n_pub)]
    d = pd.DataFrame(X, columns=[f'F{j}' for j in range(n_feat)])
    d.insert(0, 'compound', comp)
    d['smiles'] = 'CCO'
    d['label'] = label
    d['source'] = ['internal'] * n_int + ['EXP'] * n_pub
    d['origin'] = ['internal'] * n_int + ['ChEMBL'] * n_pub
    d['_ik'] = 'IK'
    int_ids = np.array(comp[:n_int])
    pub_ids = comp[n_int:]
    # 5-fold CV over internal; public added to TRAIN only (never a test fold) — mirrors select_best_combo_and_update
    from sklearn.model_selection import KFold
    folds = [[list(int_ids[tr]), list(int_ids[te])] for tr, te in KFold(5, shuffle=True, random_state=42).split(int_ids)]
    fold_ids_aug = [[tr + pub_ids, te] for tr, te in folds]
    return types.SimpleNamespace(d=d, int_ids=int_ids, pub_ids=pub_ids, combo=['ChEMBL'], fold_ids_aug=fold_ids_aug)


class _FakeRegistry:
    """Captures the registry.add(...) call instead of writing a vault; returns a fixed model id."""
    def __init__(self):
        self.calls = []
    def add(self, **kw):
        self.calls.append(kw)
        return 99


def test_deploy_dry_run_calibration_and_fit():
    """dry_run: fits internal+public, calibrates conf_recal, returns the params; no registration."""
    params, data = _fake_params(), _fake_data()
    out = OUTPUT(params)
    s = out.deploy_endpoint(data, params, registry=None, k='sol', dry_run=True)
    # the 4 conf_recal calibration scalars are present and finite
    for kk in ('rmse_cv', 'recal_a', 'recal_b', 'label_std'):
        assert np.isfinite(s['calibration'][kk])
    # trained on internal + selected public, and nothing was registered
    assert s['n_train'] == len(data.int_ids) + len(data.pub_ids)
    assert s['model_id'] is None
    # a positive CV rmse was measured on the augmented folds
    assert np.isfinite(s['cv_rmse']) and s['cv_rmse'] > 0


def test_deploy_registers_new_h237_model_with_calibration():
    """With a registry: registers adme_sol_h237 (features_type H237), bundling model+feature_cols+calibration."""
    params, data = _fake_params(), _fake_data()
    reg = _FakeRegistry()
    s = OUTPUT(params).deploy_endpoint(data, params, registry=reg, k='sol', dry_run=False)
    # one registration happened and the returned id matches
    assert s['model_id'] == 99 and len(reg.calls) == 1
    kw = reg.calls[0]
    # a NEW model: adme_<k><suffix>, H237, single-task sklearn regression
    assert kw['experiment_name'] == 'adme_sol_h237'
    assert kw['features_type'] == 'H237' and kw['framework'] == 'sklearn'
    assert kw['model_type'] == 'single_task_regression'
    # the artifact bundle is self-contained: fitted model + trained columns + conf_recal calibration
    bundle = kw['model']
    assert set(bundle['feature_cols']) == {f'F{j}' for j in range(8)}
    assert bundle['features'] == 'H237' and bundle['sources'] == ['internal', 'ChEMBL']
    for kk in ('rmse_cv', 'recal_a', 'recal_b', 'label_std'):
        assert kk in bundle['calibration']
    assert hasattr(bundle['model'], 'predict')            # a fitted estimator
    # recal scalars mirrored into metrics for details()/trail without loading the artifact
    for kk in ('rmse_cv', 'recal_a', 'recal_b'):
        assert kk in kw['metrics']
    # the archived training set is sliced to the canonical [compound, smiles, label]
    assert list(kw['training_set'].columns) == ['compound', 'smiles', 'label']
    assert len(kw['training_set']) == len(data.int_ids) + len(data.pub_ids)


# ---- the deployment recipe: 5 CV fits for the calibration + 1 shipped fit ----

class _CountingRF(RandomForestRegressor):
    """A champion RF that records every fit, so a test can count them and see each train size.

    K_fold_by_defined_IDs deepcopies the model per fold, so the counters live on the CLASS.
    """
    n_fits = 0
    train_sizes = []

    def fit(self, X, y, **kw):
        _CountingRF.n_fits += 1
        _CountingRF.train_sizes.append(len(y))
        return super().fit(X, y, **kw)


def _deploy_with_capture(n_int=60, n_pub=200):
    """Run deploy_endpoint with a counting model, capturing the CV frame ML_Reg was handed and returned."""
    params, data = _fake_params(), _fake_data(n_int=n_int, n_pub=n_pub)
    _CountingRF.n_fits, _CountingRF.train_sizes = 0, []
    out = OUTPUT(params)
    out.make_model = lambda *a, **kw: _CountingRF(**params.RF_SINGLETASK['champion'],
                                                  n_jobs=1, random_state=params.RF_SINGLETASK['seed'])
    cap = {}
    real = m.ML_Reg.K_fold_by_defined_IDs

    def _spy(df, ID, ID_sets, **kw):
        cap['id_sets'] = ID_sets
        res = real(df, ID, ID_sets, **kw)
        cap['cv'] = res[1]
        return res

    m.ML_Reg.K_fold_by_defined_IDs = _spy
    try:
        s = out.deploy_endpoint(data, params, registry=None, k='sol', dry_run=True)
    finally:
        m.ML_Reg.K_fold_by_defined_IDs = real
    return s, data, cap


def test_deployment_does_five_cv_fits_then_one_shipped_fit():
    """Expect: exactly 6 fits — 5 calibration folds (80% internal + ALL public) then 1 shipped fit
    (100% internal + ALL public). This is the recipe documented in deploy_endpoint."""
    s, data, cap = _deploy_with_capture()
    n_int, n_pub = len(data.int_ids), len(data.pub_ids)
    # 5 folds for the calibration plus the single deployable model
    assert _CountingRF.n_fits == 6, _CountingRF.n_fits
    # each of the 5 fold fits sees ~4/5 of the internal rows PLUS every public row
    for sz in _CountingRF.train_sizes[:5]:
        assert n_pub + 0.75 * n_int <= sz <= n_pub + 0.85 * n_int, sz
    # the shipped fit is the only one that sees every internal row
    assert _CountingRF.train_sizes[5] == n_int + n_pub, _CountingRF.train_sizes[5]
    assert s['n_train'] == n_int + n_pub


def test_calibration_uses_every_internal_compound_exactly_once_and_no_public():
    """Expect: the calibration frame is the union of the 5 validation arms — one row per internal
    compound, none left out, and NOT a single public compound (public sits in every train block)."""
    s, data, cap = _deploy_with_capture()
    cv = cap['cv']
    # every internal compound contributes exactly one held-out row: nothing is set aside, nothing repeats
    assert len(cv) == len(data.int_ids), (len(cv), len(data.int_ids))
    assert cv['compound'].nunique() == len(data.int_ids)
    assert set(cv['compound']) == set(data.int_ids)
    # public compounds never enter a test fold, so they contribute no calibration row
    assert not set(cv['compound']) & set(data.pub_ids)
    # and the fold ids the CV was handed put the public rows in TRAIN only
    for tr, te in cap['id_sets']:
        assert set(data.pub_ids) <= set(tr) and not (set(te) & set(data.pub_ids))
    # the 5 test blocks partition the internal set
    assert sorted(cv['fold'].unique()) == [1, 2, 3, 4, 5]


def test_bundled_calibration_is_recomputable_from_the_validation_arms():
    """Expect: the 3 numbers shipped in the bundle are exactly what calibrate_confidence_params gives
    when re-run on the captured CV frame — proving the calibration comes from the held-out folds."""
    s, data, cap = _deploy_with_capture()
    again = m.ML_Reg.calibrate_confidence_params(cap['cv'])
    for kk in ('rmse_cv', 'recal_a', 'recal_b', 'label_std'):
        # bit-for-bit: the deployed calibration IS the fit on the validation arms
        assert abs(s['calibration'][kk] - float(again[kk])) < 1e-12, (kk, s['calibration'][kk], again[kk])


def test_webapp_applies_the_same_confidence_formula_as_the_deployed_calibration():
    """Expect: webapp._confidence(std, bundle) == exp(-clip(recal_a + recal_b*std, 0) / rmse_cv),
    so the number a user sees is the deployed calibration applied to the live tree spread."""
    from webapp.app import _confidence
    s, data, cap = _deploy_with_capture()
    c = s['calibration']
    std = np.linspace(0.0, 1.5, 25)
    want = np.exp(-np.clip(c['recal_a'] + c['recal_b'] * std, 0.0, None) / c['rmse_cv'])
    got = _confidence(std, {'calibration': c, 'sigma': None})
    # the webapp must not re-derive or rescale anything
    assert np.allclose(got, want), np.abs(got - want).max()
    # a confidence is a probability-like score in (0, 1]
    assert (got > 0).all() and (got <= 1.0 + 1e-12).all()
