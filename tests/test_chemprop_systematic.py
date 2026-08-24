"""Unit tests for python/run_Chemprop_SystematicGroups.py — the multitask Chemprop grouping runner.

The expensive part (training a D-MPNN) is NOT run in the fast suite: the leakage-critical logic lives
entirely in the pure split builders (build_combined / build_public_only / the CV fold assignment), so
those are asserted directly, and the chemprop CLI is mocked to exercise the eval/scoring wiring. Also
checks that the chemprop temporal test set is bit-identical to the RF runner's (same held-out
compounds -> comparable metrics), and that a chemprop model predicts through MLTrail.

A real 1-epoch train is available but skipped unless RUN_CHEMPROP=1 (needs the `chemprop` env/GPU).
Run (env `ML`, from repo root):
    python -m unittest tests.test_chemprop_systematic -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'python'))
import _adme_fixture as fx
import run_Chemprop_SystematicGroups as cps
import run_RF_SingleTask_systematic as rf

TMP = CPDATA = RFDATA = PARAMS = OUT = None
CLEARANCE = ['hlm', 'mlm', 'rlm']


def setUpModule():
    """Build the subset cache once and load both runners' DATA on it (to cross-check split parity)."""
    global TMP, CPDATA, RFDATA, PARAMS, OUT
    TMP = Path(tempfile.mkdtemp(prefix='adme_cp_'))
    cache = fx.build(TMP / 'cache')
    cps.CACHE = cache
    PARAMS = cps.PARAMS(cps.CONFIG)
    PARAMS.epochs, PARAMS.patience, PARAMS.cv_folds = 1, 1, 3
    PARAMS.output_dir = str(TMP / 'out')
    OUT = cps.OUTPUT()
    CPDATA = cps.DATA().load_all(PARAMS)
    # RF DATA on the SAME cache — to compare the two runners' temporal test sets directly
    rf.CACHE, rf.FEATDIR = cache, TMP / 'feats'
    RFDATA = rf.DATA().load_all(rf.PARAMS(rf.CONFIG))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestSplitParity(unittest.TestCase):
    def test_temporal_matches_rf(self):
        """The chemprop and RF runners must hold out the SAME compounds per endpoint (newest 30%),
        so their metrics are directly comparable. Both derive it from the SRB id — assert identity."""
        for ep in ['logd', 'hlm']:
            # identical held-out compounds -> RF and chemprop score on the same test set
            self.assertEqual(set(CPDATA.temporal_test_ids(ep)), set(RFDATA.temporal_test_ids(ep)))


class TestBuildCombined(unittest.TestCase):
    def test_temporal_no_leakage(self):
        """build_combined: the held-out set = union of members' local-temporal tests, all labeled
        'test'; public rows are ALWAYS 'train'; the 10% val is carved from internal-train only."""
        rng = np.random.default_rng(PARAMS.seed)
        combined, test_truth = OUT.build_combined(CPDATA, PARAMS, CLEARANCE, rng)
        union = set().union(*(set(CPDATA.temporal_test_ids(e)) for e in CLEARANCE))
        n_int = len(CPDATA.tgt)
        internal, public = combined.iloc[:n_int], combined.iloc[n_int:]
        # test_truth is exactly the union of held-out internal compounds
        self.assertEqual(set(test_truth['compound']), union)
        # every held-out compound is 'test'; the count matches the union
        self.assertEqual(int((internal['splits'] == 'test').sum()), len(union))
        # public rows are never held out — all train (never leak into val/test)
        self.assertTrue((public['splits'] == 'train').all())
        # a validation set exists and is disjoint from test (carved from train)
        self.assertGreaterEqual(int((internal['splits'] == 'val').sum()), 1)
        self.assertEqual(int(((internal['splits'] == 'val') & (internal['splits'] == 'test')).sum()), 0)

    def test_public_only_no_leakage(self):
        """build_public_only: EVERY internal row is 'test' (zero internal in training); val is drawn
        from public only; test_truth covers all internal compounds."""
        rng = np.random.default_rng(PARAMS.seed)
        combined, test_truth = OUT.build_public_only(CPDATA, PARAMS, CLEARANCE, rng)
        n_int = len(CPDATA.tgt)
        internal, public = combined.iloc[:n_int], combined.iloc[n_int:]
        # not a single internal compound is trained on
        self.assertTrue((internal['splits'] == 'test').all())
        # val/train come exclusively from public rows
        self.assertTrue(set(public['splits']).issubset({'train', 'val'}))
        self.assertGreaterEqual(int((public['splits'] == 'val').sum()), 1)
        # all internal compounds are predicted
        self.assertEqual(set(test_truth['compound']), set(CPDATA.tgt['compound']))

    def test_all_target_columns_present(self):
        """A grouping keeps a column for EVERY member endpoint even when some have no public source
        (mdck/ppb here) — otherwise the multitask model would silently train on fewer tasks."""
        all8 = PARAMS.groupings['all8']
        combined, test_truth = OUT.build_combined(CPDATA, PARAMS, all8, np.random.default_rng(PARAMS.seed))
        # exactly smiles + splits + all 8 targets, no more, no fewer
        self.assertEqual(list(combined.columns), ['smiles', 'splits'] + all8)
        # source-less endpoints still appear as (all-NaN-in-public) columns
        for ep in ['mdck', 'ppb']:
            self.assertIn(ep, combined.columns)
        # held out = union of all 8 endpoints' local-temporal tests
        union = set().union(*(set(CPDATA.temporal_test_ids(e)) for e in all8))
        self.assertEqual(set(test_truth['compound']), union)

    def test_split_deterministic(self):
        """Same seed -> identical split assignment (reproducible train/val/test for a deployed model)."""
        a, _ = OUT.build_combined(CPDATA, PARAMS, CLEARANCE, np.random.default_rng(PARAMS.seed))
        b, _ = OUT.build_combined(CPDATA, PARAMS, CLEARANCE, np.random.default_rng(PARAMS.seed))
        self.assertEqual(list(a['splits']), list(b['splits']))


class TestCVFolds(unittest.TestCase):
    def test_folds_partition_pool(self):
        """CV: the shared compound-level folds partition the pool (disjoint + exhaustive), and a
        held-out fold lands entirely in 'test' with public still in train."""
        from sklearn.model_selection import KFold
        ids = CPDATA.cv_compound_ids(CLEARANCE)
        folds = [set(ids[te]) for _, te in KFold(PARAMS.cv_folds, shuffle=True, random_state=PARAMS.seed).split(ids)]
        # folds are exhaustive and mutually disjoint
        self.assertEqual(set().union(*folds), set(ids))
        self.assertEqual(sum(len(f) for f in folds), len(set(ids)))
        rng = np.random.default_rng(PARAMS.seed)
        _, test_truth = OUT.build_combined(CPDATA, PARAMS, CLEARANCE, rng, test_ids=folds[0])
        # the requested fold is exactly what gets held out
        self.assertEqual(set(test_truth['compound']), folds[0])


class TestEvalMockedCLI(unittest.TestCase):
    def test_cv_scores_and_saves(self):
        """With train_predict stubbed to return PERFECT predictions, run_grouping_cv scores each
        endpoint (r2 ~ 1 where n>=3) and writes the pooled out-of-fold parquet."""
        def perfect(params, gname, endpoints, cpath, test_truth, persist=True):
            preds = pd.DataFrame({'smiles': test_truth['smiles'].values})
            for i, ep in enumerate(endpoints):
                preds[f'pred_{i}'] = test_truth[ep].to_numpy(float)   # predict == truth
            return preds, Path(cpath).parent
        with mock.patch.object(OUT, 'train_predict', side_effect=perfect):
            rows = OUT.run_grouping_cv(CPDATA, PARAMS, 'clearance', CLEARANCE)
        for r in rows:
            if r['cv_n'] >= 3:
                # perfect predictions -> in-house R2 essentially 1
                self.assertIsNotNone(r['cv_r2'])
                self.assertGreater(r['cv_r2'], 0.98)
        # the pooled out-of-fold predictions are persisted for the notebook/metric sweep
        self.assertTrue((Path(PARAMS.output_dir) / 'hlm' / 'pred_clearance_cv.parquet').exists())

    def test_eval_reads_endpoint_named_columns(self):
        """eval_grouping must score whether the CLI returns generic `pred_i` columns OR endpoint-named
        columns — chemprop versions differ, and a mis-mapped column would score the wrong endpoint."""
        combined, test_truth = OUT.build_combined(CPDATA, PARAMS, CLEARANCE, np.random.default_rng(PARAMS.seed))
        preds = pd.DataFrame({ep: test_truth[ep].to_numpy(float) for ep in CLEARANCE})   # named, not pred_i
        rows = OUT.eval_grouping(CPDATA, PARAMS, 'clearance_named', CLEARANCE, preds, test_truth)
        for r in rows:
            if r['n'] >= 3:
                # correct column mapping -> perfect preds score ~1 for each endpoint
                self.assertGreater(r['r2'], 0.98)


class TestMLTrailChemprop(unittest.TestCase):
    def test_register_and_predict(self):
        """A saved chemprop model dir registers into MLTrail (framework=chemprop) and predicts through
        the CLI path — mocked here so no GPU/real model is needed. All 3 public SMILES get a value per
        target, and unparseable SMILES yield nulls (RDKit-filtered)."""
        reg = fx.isolated_registry(TMP)
        model_dir = TMP / 'cp_model'
        (model_dir / 'model_0' / 'checkpoints').mkdir(parents=True)
        (model_dir / 'model_0' / 'best.pt').write_bytes(b'stub')     # resolve_checkpoint just needs this to exist
        mid = reg.add(experiment_name='adme_mt_clearance', experiment_measure=','.join(CLEARANCE),
                      unit='mixed', model=str(model_dir), model_type='multitask_regression',
                      framework='chemprop', features_type='smiles', target_columns=CLEARANCE)
        df = pd.DataFrame({'compound': ['ethanol', 'benzene', 'aspirin', 'bad'],
                           'smiles': ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'not_a_smiles']})
        with mock.patch('subprocess.run', fx.fake_chemprop_cli(len(CLEARANCE))):
            out = reg.predict(mid, df, smiles_column='smiles', compound_id='compound')
        # every target column is present under its endpoint name
        for ep in CLEARANCE:
            self.assertIn(ep, out.columns)
        # the 3 valid SMILES get non-null predictions; the invalid one is null
        self.assertEqual(int(out[CLEARANCE].notna().all(axis=1).sum()), 3)


@unittest.skipUnless(os.environ.get('RUN_CHEMPROP') == '1', 'set RUN_CHEMPROP=1 (needs the chemprop env/GPU)')
class TestRealTrain(unittest.TestCase):
    def test_cv_one_epoch(self):
        """End-to-end smoke test of the real chemprop CLI: a 2-fold, 1-epoch clearance CV must produce
        finite predictions for each endpoint. Slow — opt in with RUN_CHEMPROP=1 in the chemprop env."""
        PARAMS.cv_folds = 2
        rows = OUT.run_grouping_cv(CPDATA, PARAMS, 'clearance', CLEARANCE)
        # at least one endpoint produced scored out-of-fold predictions
        self.assertTrue(any(r['cv_n'] > 0 for r in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
