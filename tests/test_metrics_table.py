"""Tests for fn.endpoint_metrics_table (moved from the FUNCTIONS cell of Multitask_adme_preds).

Synthetic arm stores only: random real_y / pred_y with fake C_ ids. Run with
`python -m unittest discover -s tests` (env ML, repo root).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))]
import IPython.display as ipd
import python.functions as fn
from python.ADME_build_ML import PARAMS

PARAMS_ = PARAMS(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')).load_params()


def _store(arms, n=60, seed=0):
    """Arm store shaped like output.metrics_results[k]: <arm>_preddf + <arm>_metrics, plus _calibration."""
    rng, r = np.random.default_rng(seed), {'_calibration': {}}
    for a in arms:
        y = rng.normal(1, 1, n)
        r[a + '_preddf'] = pd.DataFrame({'compound': [f'C_{i:03d}' for i in range(n)], 'real_y': y,
                                         'pred_y': y + rng.normal(0, .5, n)})
        r[a + '_metrics'] = {'r2det': rng.random(), 'r2': rng.random(), 'rmse': rng.random(),
                             'n_test': n, 'n_train': 5 * n}
    return r


def _table(store, ep):
    """Run the helper with display() captured; return (frame, html)."""
    shown, keep = [], ipd.display
    ipd.display = lambda h: shown.append(h.data)
    try:
        return fn.endpoint_metrics_table(store, ep, PARAMS_), shown[0]
    finally:
        ipd.display = keep


class TestMetricsTable(unittest.TestCase):
    """The per-endpoint metrics table helper."""

    def test_rows_are_cv_first_then_temporal(self):
        """Input: 2 CV arms + 2 temporal arms. Expect: one row per arm, the CV rows before the temporal
        rows, R2det descending inside each block, and no row for _calibration."""
        df, _ = _table(_store(['internal_cv', 'internal_temp', 'augmented_cv', 'augmented_temp']), 'hlm')
        # one row per arm, nothing for _calibration
        assert sorted(df.index) == ['augmented_cv', 'augmented_temp', 'internal_cv', 'internal_temp']
        # CV block first
        assert [('temp' in a) for a in df.index] == [False, False, True, True]
        # R2det descending inside each block
        assert df['R2_det'].iloc[0] >= df['R2_det'].iloc[1] and df['R2_det'].iloc[2] >= df['R2_det'].iloc[3]

    def test_triage_uses_the_config_cutoff_and_side(self):
        """Input: one arm where every real and predicted value sits on the favorable side of the config
        cutoff. Expect: Accuracy 1 and PPV 1, so the cutoff and its side come from webapp.cutoffs."""
        ep = 'logd'
        cut = PARAMS_.webapp['cutoffs'][ep]
        s = _store(['internal_cv'])
        s['internal_cv_preddf'][['real_y', 'pred_y']] = cut['value'] - 1.0 if cut['favorable'] == '<' else cut['value'] + 1.0
        df, html = _table(s, ep)
        # all compounds land in the positive class on both sides
        assert df.loc['internal_cv', 'Accuracy'] == 1.0 and df.loc['internal_cv', 'PPV'] == 1.0
        # the caption names the config cutoff
        assert f'cutoff {cut["value"]:g}' in html


if __name__ == '__main__':
    unittest.main()
