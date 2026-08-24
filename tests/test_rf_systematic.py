"""Unit tests for python/run_RF_SingleTask_systematic.py — the single-task RandomForest ADME runner.

Runs the champion RF (tiny hyperparams) on a ~1K-compound LOCAL subset of the real cache and checks
the properties that matter: features load, and — the point of the suite — that NO arm leaks the
held-out compounds into training (temporal, 5-fold CV, and public-only), plus that a registered model
predicts end-to-end through MLTrail.

Split: setUpModule builds the fixture + loads DATA once (expensive); each test only asserts, so the
asserts can be tweaked without re-fitting. Run (env `ML`, from repo root):
    python -m unittest tests.test_rf_systematic -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # _adme_fixture
sys.path.insert(0, str(HERE.parent / 'python'))     # the runner under test
import _adme_fixture as fx
import run_RF_SingleTask_systematic as rf

TMP = DATA = PARAMS = OUT = None
EP = 'logd'                                          # a tested endpoint (has a public-EXP subset)


def setUpModule():
    """Build the subset cache, point the runner's module globals at it, and load DATA once."""
    global TMP, DATA, PARAMS, OUT
    TMP = Path(tempfile.mkdtemp(prefix='adme_rf_'))
    rf.CACHE = fx.build(TMP / 'cache')              # runner reads these module globals at call time
    rf.FEATDIR = TMP / 'feats'
    PARAMS = rf.PARAMS(rf.CONFIG)
    PARAMS.endpoints = ['logd', 'hlm']
    PARAMS.champion = {'n_estimators': 15, 'max_depth': 8, 'min_samples_leaf': 2}   # tiny + fast
    PARAMS.n_jobs, PARAMS.cv_folds = 2, 3
    PARAMS.rf_kw = dict(PARAMS.champion, n_jobs=PARAMS.n_jobs, random_state=PARAMS.seed)
    PARAMS.output_dir = str(TMP / 'out')            # absolute -> `ROOT / output_dir` resolves here
    OUT = rf.OUTPUT()
    DATA = rf.DATA().load_all(PARAMS)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestData(unittest.TestCase):
    def test_features_loaded(self):
        """load_all yields the full H236 feature set (4269 cols) and a non-empty internal frame."""
        # H236 = 2048 F* + 2048 AP* + 167 MACCS* + 7 physchem
        self.assertEqual(len(DATA.feats), 4269)
        # internal targets merged with features, at least the tested endpoints present
        self.assertGreater(len(DATA.internal), 0)

    def test_internal_ep_has_label(self):
        """internal_ep(ep) renames the endpoint column to 'label' with no missing values."""
        d = DATA.internal_ep(EP)
        # every returned row is a measured compound (dropna on the endpoint)
        self.assertIn('label', d.columns)
        self.assertTrue(d['label'].notna().all())


class TestTemporalSplit(unittest.TestCase):
    def test_no_leakage(self):
        """Temporal arm: newest ~30% held out; trained and tested compounds are disjoint, and the
        model only predicts held-out compounds (never something it saw in training)."""
        aug = DATA.augmented_sources(EP, PARAMS)
        ML, d, _ = DATA.pooled(EP, aug)
        test = DATA.temporal_test_ids(EP)
        test_ids = set(d.loc[d['compound'].isin(test), 'compound'])
        train_ids = set(ML.loc[~ML['compound'].isin(test), 'compound'])
        # the split is clean: no compound is both trained on and held out
        self.assertTrue(train_ids.isdisjoint(test_ids))
        # the held-out set is ~30% of this endpoint's measured internal compounds
        self.assertTrue(0.2 <= len(test_ids) / len(d) <= 0.4)
        pred = OUT.eval_temporal(DATA, PARAMS, EP, aug)
        # predictions are made ONLY on held-out compounds
        self.assertTrue(set(pred['compound']).issubset(test_ids))


class TestCVSplit(unittest.TestCase):
    def test_no_leakage(self):
        """5-fold CV arm: public rows are always in TRAIN (never scored); pooled out-of-fold
        predictions cover every internal compound exactly once and nothing else."""
        aug = DATA.augmented_sources(EP, PARAMS)
        _, d, pub = DATA.pooled(EP, aug)
        pub_ids = set(pd.concat(pub)['compound']) if pub else set()
        internal_ids = set(d['compound'])
        pred = OUT.eval_cv(DATA, PARAMS, EP, aug)
        scored = set(pred['compound'])
        # out-of-fold predictions are internal-only — no public compound is ever in a test fold
        self.assertTrue(scored.isdisjoint(pub_ids))
        # every internal compound is predicted exactly once (a clean partition into folds)
        self.assertEqual(scored, internal_ids)
        self.assertEqual(len(pred), len(pred['compound'].unique()))


class TestPublicOnlySplit(unittest.TestCase):
    def test_no_leakage(self):
        """Public-only arm: train = public compounds, test = ALL internal. The two sets are
        compound-disjoint (zero internal leakage into training) and every internal compound is scored."""
        aug = DATA.augmented_sources(EP, PARAMS)
        _, d, pub = DATA.pooled(EP, aug)
        pub_ids = set(pd.concat(pub)['compound'])
        internal_ids = set(d['compound'])
        # training (public) shares no compound with the internal test set
        self.assertTrue(pub_ids.isdisjoint(internal_ids))
        pred = OUT.eval_public_only(DATA, PARAMS, EP, aug)
        # predicts exactly the internal compounds, none of the public training rows
        self.assertEqual(set(pred['compound']), internal_ids)


class TestMLTrail(unittest.TestCase):
    def test_register_and_predict(self):
        """run_endpoint fits + registers the deployable model into an isolated MLTrail vault, and a
        follow-up predict on public reference SMILES returns a non-null value for all 3 (the H236
        featurizer aligns end-to-end)."""
        reg = fx.isolated_registry(TMP)
        PARAMS.register_mltrail = True
        OUT.run_endpoint(DATA, PARAMS, EP, reg)
        listing = reg.list()
        # the endpoint model landed in the registry under adme_<ep>
        self.assertTrue((listing['experiment_name'] == f'adme_{EP}').any())
        mid = int(listing.loc[listing['experiment_name'] == f'adme_{EP}', 'id'].iloc[0])
        n_ok, n_tot = OUT.deploy_sanity(reg, mid)
        # all 3 public SMILES get a non-null prediction through MLTrail
        self.assertEqual((n_ok, n_tot), (3, 3))
        # the deployed model records its modelling unit so downstream consumers can invert the transform
        self.assertEqual(reg.details(mid)['unit'], 'logD7.4')
        self.assertEqual(reg.details(mid)['features_type'], 'H236')

    def test_predict_invariant_to_input_shape(self):
        """A consumer may pass columns/rows in any order (or with extra columns). MLTrail featurizes
        from SMILES and aligns to the trained columns, so predictions must be identical regardless."""
        reg = fx.isolated_registry(TMP)
        PARAMS.register_mltrail = True
        OUT.run_endpoint(DATA, PARAMS, EP, reg)
        mid = int(reg.list().pipe(lambda l: l.loc[l['experiment_name'] == f'adme_{EP}', 'id'].iloc[0]))
        ref = pd.DataFrame({'compound': ['ethanol', 'benzene', 'aspirin'],
                            'smiles': ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']})
        base = reg.predict(mid, ref, smiles_column='smiles', compound_id='compound')
        col = 'prediction' if 'prediction' in base.columns else base.select_dtypes('number').columns[-1]
        base = base.set_index('compound')[col]
        shuffled = ref.iloc[::-1].assign(junk=1)               # reversed rows + an extra column
        alt = reg.predict(mid, shuffled, smiles_column='smiles', compound_id='compound').set_index('compound')[col]
        # same compound -> same prediction, independent of row order / extra columns
        for cid in base.index:
            self.assertAlmostEqual(float(base[cid]), float(alt[cid]), places=6)

    def test_idempotent_registration(self):
        """Re-running an endpoint must UPDATE the model in place (new version, same id) — never create
        a duplicate registry entry that could fork which model production serves."""
        reg = fx.isolated_registry(TMP)
        PARAMS.register_mltrail = True
        OUT.run_endpoint(DATA, PARAMS, EP, reg)
        mid1 = int(reg.list().pipe(lambda l: l.loc[l['experiment_name'] == f'adme_{EP}', 'id'].iloc[0]))
        OUT.run_endpoint(DATA, PARAMS, EP, reg)                # register the same endpoint again
        rows = reg.list().pipe(lambda l: l[l['experiment_name'] == f'adme_{EP}'])
        # exactly one entry, same id, with a second version appended (history kept)
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows['id'].iloc[0]), mid1)
        self.assertGreaterEqual(len(reg._data['models'][str(mid1)]['versions']), 2)


class TestSourcePolicy(unittest.TestCase):
    def test_config_override_filtered_to_available(self):
        """augmented_sources honors the per-endpoint config policy, filtered to parquets that exist:
        logd (no override) -> whatever is available; mdck ([NVS], absent here) -> empty."""
        # logd has no override -> only the available EXP source
        self.assertEqual(DATA.augmented_sources('logd', PARAMS), ['EXP'])
        # mdck override is [NVS], but no Novartis parquet exists in the subset -> nothing usable
        self.assertEqual(DATA.augmented_sources('mdck', PARAMS), [])

    def test_solubility_excludes_predicted_source(self):
        """The solubility poisoning guard: even when an ADMETlab (predicted) solubility parquet is
        AVAILABLE, the config override ([EXP]) keeps it out of training."""
        adm = rf.CACHE / 'public_admetlab_solubility.parquet'
        shutil.copy(rf.CACHE / 'public_solubility.parquet', adm)   # make ADM available
        self.addCleanup(adm.unlink)
        # it is discoverable...
        self.assertIn('ADM', DATA.avail_sources('solubility'))
        # ...but the config override deliberately excludes predicted solubility
        self.assertEqual(DATA.augmented_sources('solubility', PARAMS), ['EXP'])


class TestModellingUnit(unittest.TestCase):
    def test_transform_to_unit_label(self):
        """modelling_unit maps each endpoint's transform to the label consumers see — a wrong label
        means silently mis-scaled predictions in production."""
        u = lambda ep: OUT.modelling_unit(PARAMS.endpoint_cfg[ep])
        self.assertEqual(u('logd'), 'logD7.4')                 # identity -> raw unit
        self.assertEqual(u('solubility'), 'log10(uM)')         # log10 -> wrapped
        self.assertEqual(u('hlm'), 'log10(uL/min/mg)')
        self.assertEqual(u('ppb'), 'logit(fraction_unbound)')  # logit_pct -> canonical label


class TestReproducibility(unittest.TestCase):
    def test_cv_predictions_deterministic(self):
        """The same seed must reproduce identical CV predictions — a deployed model's numbers have to
        be reproducible run-to-run (fold assignment + RF are both seeded)."""
        aug = DATA.augmented_sources(EP, PARAMS)
        p1 = OUT.eval_cv(DATA, PARAMS, EP, aug).sort_values('compound').reset_index(drop=True)
        p2 = OUT.eval_cv(DATA, PARAMS, EP, aug).sort_values('compound').reset_index(drop=True)
        # identical held-out compounds and identical predicted values
        self.assertEqual(list(p1['compound']), list(p2['compound']))
        self.assertTrue(np.allclose(p1['pred_y'], p2['pred_y']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
