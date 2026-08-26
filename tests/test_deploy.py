"""Unit tests for OUTPUT.deploy_endpoint (fit internal+public RF, calibrate conf_recal, register to MLTrail).

Synthetic data only (fake feature columns + a fake registry) — no vault write, no real MLTrail, no chemistry.
Checks: the augmented-CV calibration produces the conf_recal params, the model is fit on internal+public,
the bundle carries {model, feature_cols, calibration}, and a NEW H237 model is registered with the recal
scalars mirrored into metrics.
"""
import types
import numpy as np
import pandas as pd

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
