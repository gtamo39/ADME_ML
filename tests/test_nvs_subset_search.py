"""Unit tests for python/nvs_subset_search.NVSSubsetSearch (mechanics only, synthetic public SMILES).

No real chemistry: internal + NVS rows are built from public RDKit-example SMILES with distinct ring
topologies (so Bemis-Murcko generic scaffolds form >=2 groups) and random features whose linear signal
the tiny RF can learn. Checks that every lever runs and returns well-formed output — the selection logic,
leakage-free per-fold distance, monotone nested subsets, scaffold partition, bias/weight/nested paths.
"""
import types
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from python.nvs_subset_search import NVSSubsetSearch, _pearson_r2

# public SMILES with DISTINCT generic-scaffold topologies (6-ring, 5-ring, 7-ring, fused 6-6, acyclic)
PALETTE = ['c1ccccc1', 'C1CCCC1', 'C1CCCCCC1', 'c1ccc2ccccc2c1', 'CCO']


class _FakeOutput:
    """Supplies make_model() — a small fast champion-RF stand-in."""
    def make_model(self, use_cuml=False):
        return RandomForestRegressor(n_estimators=15, random_state=0, n_jobs=1)


def _fake_data(n_int=24, n_nvs=60, n_feat=6, seed=0):
    """Build a DATA-like namespace (d/internal/pub/combo) for endpoint 'mdck' from the SMILES palette."""
    rng = np.random.default_rng(seed)
    n = n_int + n_nvs
    X = rng.normal(size=(n, n_feat))
    label = X @ rng.normal(size=n_feat) + rng.normal(scale=0.4, size=n)      # learnable signal
    comp = [f'SRB-{i:06d}-001' for i in range(n_int)] + [f'PUB_{i}' for i in range(n_nvs)]
    smiles = [PALETTE[i % len(PALETTE)] for i in range(n)]                   # cycle topologies across both sets
    d = pd.DataFrame(X, columns=[f'F{j}' for j in range(n_feat)])
    d.insert(0, 'compound', comp)
    d['smiles'] = smiles
    d['label'] = label
    d['source'] = ['internal'] * n_int + ['EXP'] * n_nvs
    d['origin'] = ['internal'] * n_int + ['Novartis-NIBR'] * n_nvs
    d['_ik'] = [f'IK{i}' for i in range(n)]                                   # unique -> each its own fold group
    data = types.SimpleNamespace(d=d,
                                 internal=d[d.source == 'internal'].copy(),
                                 pub=d[d.source == 'EXP'].copy(),
                                 combo=['Novartis-NIBR'])
    return data


def _search():
    return NVSSubsetSearch(_fake_data(), _FakeOutput(), types.SimpleNamespace(), endpoint='mdck', n_bits=512)


def test_baseline_and_precompute_shapes():
    """baseline() returns internal-only + all-NVS R2 and the right counts; the sim matrix is (n_int x n_nvs)."""
    s = _search()
    b = s.baseline()
    # counts match the constructed sets
    assert b['n_internal'] == 24 and b['n_nvs'] == 60
    # similarity matrix orientation and range
    assert s._sim.shape == (24, 60)
    assert s._sim.min() >= 0.0 and s._sim.max() <= 1.0 + 1e-6


def test_distance_selection_is_nested_and_per_train():
    """Larger tau keeps a superset of NVS; distance is computed from ONLY the given train rows (leakage-free)."""
    s = _search()
    tr = s.int_ids[:12]
    sel_small = set(s._select_within(tr, 0.2))
    sel_big = set(s._select_within(tr, 0.9))
    # monotone: a looser threshold keeps everything the tighter one kept
    assert sel_small.issubset(sel_big)
    # distance uses ONLY the train rows of the similarity matrix (direct leakage-free-indexing check)
    rows = [s._int_pos[c] for c in tr]
    assert np.allclose(s._dist_to_train(tr), 1.0 - s._sim[rows, :].max(axis=0))
    # adding more train rows can only shrink the distance (max over more neighbours) — never grow it
    assert (s._dist_to_train(s.int_ids) <= s._dist_to_train(tr) + 1e-9).all()


def test_distance_curve_counts_monotone():
    """distance_curve: tau ascending and the median NVS kept is non-decreasing in tau."""
    s = _search()
    df = s.distance_curve(taus=[0.2, 0.4, 0.6, 0.8, 1.0])
    assert list(df['tau']) == sorted(df['tau'])
    assert (df['n_nvs_median'].diff().dropna() >= 0).all()


def test_scaffold_groups_partition_and_greedy_runs():
    """scaffold_groups partitions all NVS ids; forward greedy returns a chosen list + an R2 path."""
    s = _search()
    groups = s.scaffold_groups(max_groups=4)
    allids = [c for g in groups for c in groups[g]]
    # every NVS compound lands in exactly one group
    assert sorted(allids) == sorted(s.nvs_ids)
    res = s.scaffold_greedy(direction='forward', max_groups=4)
    assert isinstance(res['chosen'], list) and 'r2' in res and len(res['path']) >= 1


def test_bias_and_weight_and_nested_paths_run():
    """S5 bias-correction, S3 weighting, and the honest nested-CV all run and return well-formed output."""
    s = _search()
    # S5: affine-corrected labels sweep
    bc = s.biascorrect_curve(sim0=0.3, taus=[0.4, 0.8])
    assert set(bc.columns) == {'tau', 'r2'} and len(bc) == 2
    # S3: within-NVS uncertainty has one value per NVS compound; weighting sweep runs
    u = s.nvs_uncertainty(n_splits=3)
    assert u.shape == (60,)
    w = s.weighted_r2(taus=[0.4, 0.8])
    assert len(w) == 2
    # nested CV: one chosen tau per outer fold, honest R2 is a float or nan
    nd = s.nested_distance(taus=[0.4, 0.8], n_outer=3, n_inner=2)
    assert len(nd['chosen_taus']) == 3
    assert isinstance(nd['honest_r2'], float)


def test_pearson_r2_guards():
    """_pearson_r2 is nan for constant/degenerate input, finite for a real relationship."""
    assert np.isnan(_pearson_r2([1, 1, 1], [1, 2, 3]))
    assert np.isfinite(_pearson_r2([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8]))


def test_cv_preddf_and_nested_return_oof_frames():
    """cv_preddf returns a well-formed OOF pred_df (each internal compound once) whose R2 matches _cv_r2;
    nested_distance now also returns a pred_df. These frames feed endpoint_metrics_table_from_dict."""
    s = _search()
    rec = []
    pdf = s.cv_preddf(lambda tr: s._select_within(tr, 0.5), record=rec)
    # frame shape + each internal compound predicted exactly once + per-fold provenance recorded
    assert set(pdf.columns) == {'compound', 'real_y', 'pred_y', 'fold', 'residuals'}
    assert sorted(pdf['compound']) == sorted(s.int_ids)
    assert len(rec) == 5 and all(k in rec[0] for k in ('fold', 'n_int_train', 'n_nvs_added', 'test_ids'))
    # R2 from the pred_df equals the scalar path on the same selection
    assert abs(_pearson_r2(pdf['real_y'], pdf['pred_y']) - s._cv_r2(lambda tr: s._select_within(tr, 0.5))) < 1e-9
    # nested returns an OOF pred_df over all internal compounds
    nd = s.nested_distance(taus=[0.4, 0.8], n_outer=3, n_inner=2)
    assert 'preddf' in nd and sorted(nd['preddf']['compound']) == sorted(s.int_ids)


def test_bias_correction_recalibrates_to_internal_scale():
    """S5 _bias_label: 'shift' matches the internal-train median; 'affine' matches median AND IQR; empty add -> {}."""
    from python.nvs_subset_search import _iqr
    s = _search()
    tr, add = s.int_ids[:18], s.nvs_ids[:30]
    yint = s._byid.loc[tr, 'label'].to_numpy(float)
    # shift: corrected NVS labels take the internal-train median (removes a constant offset)
    cv = np.array([s._bias_label(method='shift')(tr, add)[c] for c in add])
    assert abs(np.median(cv) - np.median(yint)) < 1e-6
    # affine: corrected NVS labels match internal-train median AND spread (offset + scale)
    cv2 = np.array([s._bias_label(method='affine')(tr, add)[c] for c in add])
    assert abs(np.median(cv2) - np.median(yint)) < 1e-6 and abs(_iqr(cv2) - _iqr(yint)) < 1e-6
    # an empty added-NVS set means no correction
    assert s._bias_label('shift')(tr, []) == {}
