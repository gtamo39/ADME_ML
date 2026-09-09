"""Tests for the round-trip notebook cells of vignettes/Multitask_adme_preds.ipynb.

The cells score the webapp's own predictions on the internal set against the CDD truth. The real
files hold chemical structures, so the cells only ever run here on SYNTHETIC frames that mimic the
real column names: the raw CDD columns from params.ADME_ENDPOINTS[<ep>]['col'] and the webapp's
22-column export (<ep>_pred / <ep>_confidence). Run with the loop in the repo README, not unittest
discovery, because the module executes notebook source rather than defining classes.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path[:0] = [os.path.expanduser('~/Scripts'), os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))]
import ML_Reg
from python.ADME_build_ML import PARAMS

NB = os.path.join(os.path.dirname(__file__), '..', 'vignettes', 'Multitask_adme_preds.ipynb')
CELL_METRICS, CELL_PANELS, CELL_CV_PANELS = '01a26a07', 'eb77e860', '560d161e'
_INV = {'log10': lambda a: 10.0 ** a, 'identity': lambda a: a,
        'logit_pct': lambda a: 100.0 / (1.0 + 10.0 ** -a)}


def _cell_source(cell_id):
    """Return one notebook cell's SOURCE by id (never its outputs, which may hold structures)."""
    with open(NB) as f:
        nb = json.load(f)
    return ''.join(next(c for c in nb['cells'] if c.get('id') == cell_id)['source'])


def _fixture(tmpdir, n=40, seed=0):
    """Build a synthetic (df_internal_exp_all, prediction csv path, params, truth-in-modelling-space).

    The truth is drawn in MODELLING space with sd 1 and then inverted to raw units, so every endpoint
    gets the same signal-to-noise whatever its transform. Predictions are the truth plus small noise
    and a deliberate +0.15 offset, so `bias` must come out positive and R2_pears must exceed R2_det.
    """
    rng = np.random.default_rng(seed)
    params = PARAMS(os.path.join(os.path.dirname(NB), '..', 'config', 'config.yaml')).load_params()
    eps = params.ADME_ENDPOINTS
    raw = {'name': [f'C_{i:03d}' for i in range(n)], 'smiles': ['CCO', 'c1ccccc1'] * (n // 2)}
    truth = {}
    for k, ep in eps.items():
        tm = rng.normal({'log10': 1.0, 'identity': 2.5, 'logit_pct': -1.0}[ep['transform']], 1.0, n)
        truth[k] = tm
        raw[ep['col']] = _INV[ep['transform']](tm)
    raw['mdck_Cell line'] = ['MDR1-MDCK'] * (n - 5) + ['wild type'] * 5   # exercises the mdck filter
    raw[eps['solubility']['col']] = raw[eps['solubility']['col']].copy()
    raw[eps['solubility']['col']][:3] = 99999.0                           # exercises label_cap_raw
    raw[eps['hlm']['col']] = raw[eps['hlm']['col']].copy()
    raw[eps['hlm']['col']][3] = 0.0                                       # log10(0) -> -inf, must drop
    internal = pd.DataFrame(raw)

    pred = {'compound': internal.name, 'smiles': internal.smiles,
            'mpo': rng.random(n), 'score': rng.random(n)}
    for k, ep in eps.items():
        pred[k + '_pred'] = _INV[ep['transform']](truth[k] + rng.normal(0, .3, n) + 0.15)
        pred[k + '_confidence'] = rng.uniform(.2, 1, n)
    pred['px_activity_pred'] = rng.random(n)
    pred['px_activity_confidence'] = rng.random(n)
    csv = os.path.join(tmpdir, '20260909_pred_internal.csv')
    pd.DataFrame(pred).to_csv(csv, index=False)
    return internal, csv, params


def _run(tmpdir, cells=(CELL_METRICS,), n=40, seed=0):
    """Exec the named notebook cells on the synthetic fixture; return the resulting namespace."""
    internal, csv, params = _fixture(tmpdir, n=n, seed=seed)

    def _to_model(v, tf):                       # verbatim from notebook cell ad5857c4
        if tf == 'log10':     return np.log10(v)
        if tf == 'logit_pct': return np.log10(v / (100.0 - v))
        return v

    class _D:
        pass
    data = _D()
    data.df_internal_exp_all = internal
    ns = {'pd': pd, 'np': np, 'plt': plt, 'stats': stats, 'ML_Reg': ML_Reg, 'data': data,
          'params': params, '_to_model': _to_model, 'SERAC_C': params.SERAC_C,
          '_REG_COLS': {'R2_det': 'r2det', 'R2_pears': 'r2', 'RMSE': 'rmse',
                        'N': 'n_test', 'n_train': 'n_train'}}
    for cid in cells:
        src = _cell_source(cid).replace("'tmp/20260909_pred_internal.csv'", repr(csv))
        exec(compile(src, cid, 'exec'), ns)
    return ns


def test_metrics_cell_covers_every_endpoint():
    """Input: synthetic truth + predictions. Expect: one metrics row per configured endpoint, and a
    tidy rt_long carrying exactly the columns the panel cell reads."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td)
    # the metrics table must hold every configured endpoint, in config order
    assert list(ns['rt_metrics'].index) == list(ns['params'].ADME_ENDPOINTS), ns['rt_metrics'].index
    # rt_long is the contract between the two cells
    assert set(ns['rt_long'].columns) == {'compound', 'endpoint', 'real_y', 'pred_y', 'conf', 'resid'}
    # a NaN residual would make a panel plot nothing without any error
    assert ns['rt_long'].resid.notna().all()


def test_metrics_cell_applies_the_config_filter_and_cap():
    """Input: 5 non-MDR1 mdck rows, one hlm value of 0, three saturated solubility values.
    Expect: the filter drops 5, log10(0) drops 1, and the truth is clipped at label_cap_raw."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td)
    long, eps = ns['rt_long'], ns['params'].ADME_ENDPOINTS
    # mdck must keep only the MDR1 cell line, exactly as DATA._endpoint_dfs does
    assert (long.endpoint == 'mdck').sum() == 35, (long.endpoint == 'mdck').sum()
    # log10 of a zero raw value is -inf and must be dropped, not scored
    assert (long.endpoint == 'hlm').sum() == 39, (long.endpoint == 'hlm').sum()
    # the saturated truths must sit exactly at the cap, not above it
    assert np.isclose(long.loc[long.endpoint == 'solubility', 'real_y'].max(),
                      np.log10(eps['solubility']['label_cap_raw']))


def test_metrics_cell_separates_r2det_from_pearson():
    """Input: predictions offset by a constant +0.15 in modelling space.
    Expect: a positive bias, and R2_pears >= R2_det everywhere, because Pearson is affine-invariant
    and therefore blind to exactly that offset."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td)
    m = ns['rt_metrics']
    # solubility is excluded: its 3 clipped truths are scored against unclipped predictions on purpose
    clean = m.drop(index='solubility')
    # the injected offset must surface as a positive bias
    assert (clean['bias'] > 0).all(), clean['bias'].tolist()
    # the synthetic predictions are the truth plus small noise, so R2 must be high
    assert (clean['R2_det'] > 0.5).all(), clean['R2_det'].tolist()
    # this is the project's calibration law: Pearson cannot see the offset, R2det can
    assert (m['R2_pears'] >= m['R2_det'] - 1e-9).all(), 'pearson fell below r2det'
    # the clipped endpoint must be the worst, which proves the clip really reaches the metric
    assert m.loc['solubility', 'R2_det'] < clean['R2_det'].min()


def test_metrics_cell_names_the_value_model_per_endpoint():
    """Expect: each row records which model produced <ep>_pred, straight from webapp.value_model."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td)
    want = ns['params'].webapp['value_model']
    # the table must attribute every value to the model the webapp config actually deploys
    assert ns['rt_metrics']['value_model'].to_dict() == want, ns['rt_metrics']['value_model'].to_dict()


def test_panel_cell_draws_one_panel_per_endpoint():
    """Input: the namespace the metrics cell leaves behind.
    Expect: a 4x2 grid of 8 axes, each with a titled Spearman rho and both axis labels set."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td, cells=(CELL_METRICS, CELL_PANELS))
    fig = ns['fig']
    # 4 rows x 2 cols = one panel per endpoint, which is what the user asked for
    assert len(fig.axes) == 8, len(fig.axes)
    assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (4, 2)
    for ax, k in zip(fig.axes, ns['params'].ADME_ENDPOINTS):
        # every panel must name its endpoint and report the rank association
        assert ax.get_title().startswith(k) and 'rho=' in ax.get_title(), ax.get_title()
        # an unlabelled confidence axis would make the panel unreadable
        assert 'confidence' in ax.get_xlabel() and 'residual' in ax.get_ylabel()
        # the confidence axis must stay pinned to 0-1 so the 8 panels compare directly
        assert ax.get_xlim() == (0.0, 1.0), ax.get_xlim()
    plt.close('all')


def test_panel_cell_splits_each_panel_at_the_config_confidence_cut():
    """Expect: every panel draws one vertical line at webapp.confidence_split and annotates the RMSE
    plus n of BOTH halves, and those two n values add up to the panel's point count."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td, cells=(CELL_METRICS, CELL_PANELS))
    split = ns['params'].webapp.get('confidence_split', 0.5)
    # the cell must take the cut from config, not hardcode it
    assert ns['CSPLIT'] == split, ns['CSPLIT']
    for ax, k in zip(ns['fig'].axes, ns['params'].ADME_ENDPOINTS):
        # exactly one vertical rule, sitting at the configured cut
        xs = [ln.get_xdata()[0] for ln in ax.get_lines() if len(set(ln.get_xdata())) == 1]
        assert xs == [split], (k, xs)
        # both halves must be annotated, so neither side is left unscored
        labels = [t.get_text() for t in ax.texts]
        assert len(labels) == 2, (k, labels)
        d = ns['rt_long'][ns['rt_long'].endpoint == k]
        want = [int((d.conf < split).sum()), int((d.conf >= split).sum())]
        got = [int(t.split('n=')[1]) if 'n=' in t else 0 for t in labels]
        # the two halves must partition the panel's points exactly
        assert got == want, (k, got, want)
        assert sum(got) == len(d), (k, sum(got), len(d))
    plt.close('all')


def test_panel_halves_report_the_rmse_of_their_own_points():
    """Expect: each half's printed RMSE equals sqrt(mean(resid^2)) over exactly that half's rows."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        ns = _run(td, cells=(CELL_METRICS, CELL_PANELS))
    split = ns['CSPLIT']
    for ax, k in zip(ns['fig'].axes, ns['params'].ADME_ENDPOINTS):
        d = ns['rt_long'][ns['rt_long'].endpoint == k]
        for t, mask in zip(ax.texts, [d.conf < split, d.conf >= split]):
            if 'RMSE' not in t.get_text():
                continue
            r = d.resid[mask]
            # the annotation must be the RMSE of its own half, to 3 decimals
            assert abs(float(t.get_text().split()[1]) - np.sqrt(np.mean(r ** 2))) < 5e-4, (k, t.get_text())
    plt.close('all')


def _cv_fixture(tmpdir, n=90, seed=1, rlm_flat=True):
    """Pickle one synthetic <ep>.pkl per endpoint into tmpdir, shaped like OUTPUT.assess_predictions.

    Each augmented_cv_preddf carries the real column set (compound, pred_y, real_y, uq_std, fold,
    residuals, conf_*). conf_recal is built so a HIGHER confidence goes with a SMALLER residual,
    except for rlm when rlm_flat, which gets a random confidence — the small-n case where the
    std/error association is not measurable.
    """
    import pickle as pkl
    rng = np.random.default_rng(seed)
    params = PARAMS(os.path.join(os.path.dirname(NB), '..', 'config', 'config.yaml')).load_params()
    for k in params.ADME_ENDPOINTS:
        m = 39 if k == 'rlm' else n
        std = rng.uniform(.1, .9, m)
        resid = np.abs(rng.normal(0, 1, m) * std)                  # error grows with the tree spread
        conf = rng.uniform(0, 1, m) if (k == 'rlm' and rlm_flat) else np.exp(-std)
        pdf = pd.DataFrame({'compound': [f'C_{i:03d}' for i in range(m)],
                            'pred_y': rng.normal(0, 1, m), 'real_y': rng.normal(0, 1, m),
                            'uq_std': std, 'fold': rng.integers(1, 6, m), 'residuals': resid,
                            'conf_labelstd': conf, 'conf_rmse': conf,
                            'conf_recal': conf, 'conf_conformal': conf})
        with open(os.path.join(tmpdir, k + '.pkl'), 'wb') as f:
            pkl.dump({k: {'augmented_cv_preddf': pdf,
                          '_calibration': {'rmse_cv': 1.0, 'recal_a': 0.0, 'recal_b': 1.0}}}, f)
    params.METRICS_PKL_RF_DIR = tmpdir
    return params


def _run_cv_panel(tmpdir, **kw):
    """Exec the CV-panel cell against synthetic pickles; return its namespace."""
    import pickle as pkl
    params = _cv_fixture(tmpdir, **kw)

    class _O:
        pass
    out = _O()
    out.metrics_results = {}
    ns = {'pd': pd, 'np': np, 'plt': plt, 'stats': stats, 'os': os, 'pickle': pkl,
          'params': params, 'output': out, 'SERAC_C': params.SERAC_C}
    exec(compile(_cell_source(CELL_CV_PANELS), CELL_CV_PANELS, 'exec'), ns)
    return ns


def test_cv_panel_loads_every_endpoint_from_the_pickles():
    """Input: one synthetic pickle per endpoint, and an EMPTY output.metrics_results.
    Expect: the cell tops up all 8 from METRICS_PKL_RF_DIR, so the panel never depends on which
    endpoint happened to run last."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        ns = _run_cv_panel(td)
    # the top-up must fill every configured endpoint
    assert sorted(ns['output'].metrics_results) == sorted(ns['params'].ADME_ENDPOINTS)
    # one panel per endpoint, in a 4x2 grid
    assert len(ns['fig'].axes) == 8, len(ns['fig'].axes)
    plt.close('all')


def test_cv_panel_splits_and_scores_both_halves():
    """Expect: a rule at webapp.confidence_split, both halves annotated, and each printed RMSE equal
    to sqrt(mean(residuals^2)) over exactly that half of that endpoint's pred_df."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        ns = _run_cv_panel(td)
    split = ns['CSPLIT']
    for ax, k in zip(ns['fig'].axes, ns['params'].ADME_ENDPOINTS):
        d = ns['output'].metrics_results[k]['augmented_cv_preddf']
        # exactly one vertical rule, at the configured cut
        xs = [ln.get_xdata()[0] for ln in ax.get_lines() if len(set(ln.get_xdata())) == 1]
        assert xs == [split], (k, xs)
        for t, mask in zip(ax.texts, [d.conf_recal < split, d.conf_recal >= split]):
            res = d.residuals[mask]
            # the two n values must partition the endpoint's rows
            assert int(t.get_text().split('n=')[1]) == len(res), (k, t.get_text(), len(res))
            # and the RMSE must be that half's own
            assert abs(float(t.get_text().split()[1]) - np.sqrt(np.mean(res ** 2))) < 5e-4, (k, t.get_text())
    plt.close('all')


def test_cv_panel_skips_an_endpoint_with_no_pickle():
    """Input: the solubility pickle deleted. Expect: that panel says 'not available' and is switched
    off, and the other 7 still draw — a missing arm must not raise."""
    import tempfile
    plt.close('all')
    with tempfile.TemporaryDirectory() as td:
        params = _cv_fixture(td)
        os.remove(os.path.join(td, 'solubility.pkl'))
        import pickle as pkl

        class _O:
            pass
        out = _O()
        out.metrics_results = {}
        ns = {'pd': pd, 'np': np, 'plt': plt, 'stats': stats, 'os': os, 'pickle': pkl,
              'params': params, 'output': out, 'SERAC_C': params.SERAC_C}
        exec(compile(_cell_source(CELL_CV_PANELS), CELL_CV_PANELS, 'exec'), ns)
    # the endpoint with no pickle must be reported, not silently skipped
    assert 'solubility' not in ns['output'].metrics_results
    assert 'not available' in ns['fig'].axes[0].get_title(), ns['fig'].axes[0].get_title()
    # the remaining 7 must still have drawn their split
    assert sum('rho=' in a.get_title() for a in ns['fig'].axes) == 7
    plt.close('all')
