"""Unit tests for webapp/app.py: the per-run store behind the SDF export, and the startup model check.

Synthetic rows only — public reference SMILES (ethanol, benzene) and fake predictions. No vault, no
models, no uvicorn: the route function is called directly. Run: python -m unittest tests.test_webapp
"""
import asyncio
import tempfile
import time
import unittest

import pandas as pd
from fastapi import HTTPException

import webapp.app as A


def _flat(smiles, prefix):
    """A scored-run frame shaped like the one _predict_iter builds, with fake predictions."""
    return pd.DataFrame({'compound': [f'{prefix}_{i:03d}' for i in range(len(smiles))], 'smiles': smiles,
                         'logd_pred': 1.0, 'logd_confidence': 0.5})


class TestRunStore(unittest.TestCase):

    def setUp(self):
        A.RUNS.clear()
        self.webapp = dict(A.WEBAPP)

    def tearDown(self):
        A.RUNS.clear()
        A.WEBAPP.clear()
        A.WEBAPP.update(self.webapp)

    def test_each_run_gets_its_own_random_id(self):
        """Input: two stored runs. Expect: two different ids of 20+ characters, each returning its own
        frame. Rationale: the id is the only link between a download and the person who scored the file."""
        a = A._store_run(_flat(['CCO'], 'A'), [None])
        b = A._store_run(_flat(['c1ccccc1'], 'B'), [None])
        # two runs, two ids
        self.assertNotEqual(a, b)
        # too long to guess (token_urlsafe(16) gives 22 characters)
        self.assertGreaterEqual(len(a), 20)
        # each id returns its own run
        self.assertEqual(A._get_run(b)['flat'].compound.tolist(), ['B_000'])

    def test_unknown_or_absent_id_is_404(self):
        """Input: a made-up id, and no id. Expect: HTTP 404 for both while a real run is stored."""
        A._store_run(_flat(['CCO'], 'A'), [None])
        for bad in ('not-a-run', None):
            with self.assertRaises(HTTPException) as cm:
                A._get_run(bad)
            # a wrong or absent id never reaches the stored run
            self.assertEqual(cm.exception.status_code, 404)

    def test_expired_run_is_404_then_evicted(self):
        """Input: a run older than run_ttl_min. Expect: 404 on read, and gone after the next store."""
        old = A._store_run(_flat(['CCO'], 'A'), [None])
        A.RUNS[old]['t'] = time.time() - 60 * (float(A.WEBAPP.get('run_ttl_min', 60)) + 1)
        # an expired run cannot be read
        with self.assertRaises(HTTPException):
            A._get_run(old)
        A._store_run(_flat(['c1ccccc1'], 'B'), [None])
        # the next store drops it
        self.assertNotIn(old, A.RUNS)

    def test_store_holds_at_most_max_runs(self):
        """Input: 5 runs with max_runs=3. Expect: the 3 newest kept, in order; the 2 oldest dropped."""
        A.WEBAPP['max_runs'] = 3
        ids = [A._store_run(_flat(['CCO'], 'A'), [None]) for _ in range(5)]
        # the cap holds and the newest survive
        self.assertEqual(list(A.RUNS), ids[2:])


class TestSdfExport(unittest.TestCase):

    def setUp(self):
        A.RUNS.clear()

    def tearDown(self):
        A.RUNS.clear()

    def _sdf(self, payload):
        """Call the route function directly; return the SDF text."""
        return asyncio.run(A.download_sdf(payload)).body.decode()

    def test_export_returns_only_the_named_run(self):
        """Input: run A (ethanol) then run B (benzene); export row 0 of run A. Expect: one record, run A's
        compound, none of run B's — the export follows the id, not the most recent run."""
        a = A._store_run(_flat(['CCO'], 'A'), [None])
        A._store_run(_flat(['c1ccccc1'], 'B'), [None])
        sdf = self._sdf({'run': a, 'rows': [{'idx': 0}]})
        # one record
        self.assertEqual(sdf.count('$$$$'), 1)
        # from run A only
        self.assertIn('A_000', sdf)
        self.assertNotIn('B_000', sdf)

    def test_export_without_a_run_id_is_404(self):
        """Input: rows but no run id, while a run is stored. Expect: HTTP 404."""
        A._store_run(_flat(['CCO'], 'A'), [None])
        with self.assertRaises(HTTPException) as cm:
            self._sdf({'rows': [{'idx': 0}]})
        # no run id, no export
        self.assertEqual(cm.exception.status_code, 404)


class TestStartupCheck(unittest.TestCase):

    def setUp(self):
        self.saved = (dict(A.CHAMPIONS), dict(A.CP_VALUES), A.CONFIG['MLTRAIL_VAULT'])

    def tearDown(self):
        for d, s in zip((A.CHAMPIONS, A.CP_VALUES), self.saved[:2]):
            d.clear()
            d.update(s)
        A.CONFIG['MLTRAIL_VAULT'] = self.saved[2]

    def test_missing_models_names_each_gap(self):
        """Input: RF loaded for every endpoint but the first; no chemprop model. Expect: that endpoint,
        plus every endpoint that value_model sends to chemprop."""
        eps = list(A.ENDPOINT_CFG)
        A.CHAMPIONS.clear()
        A.CHAMPIONS.update({ep: {} for ep in eps[1:]})
        A.CP_VALUES.clear()
        want = [eps[0]] + [f'{ep} (chemprop)' for ep, m in A.VALUE_MODEL.items() if m == 'chemprop']
        # one RF gap plus every chemprop gap
        self.assertEqual(A._missing_models(), want)

    def test_nothing_missing_when_all_loaded(self):
        """Input: every RF model and every chemprop value model loaded. Expect: an empty list."""
        A.CHAMPIONS.update({ep: {} for ep in A.ENDPOINT_CFG})
        A.CP_VALUES.update({ep: {} for ep, m in A.VALUE_MODEL.items() if m == 'chemprop'})
        # a full load passes the strict startup check
        self.assertEqual(A._missing_models(), [])

    def test_registry_refuses_a_vault_with_no_registry(self):
        """Input: MLTRAIL_VAULT naming an empty folder. Expect: FileNotFoundError, not an empty registry
        that would start the app with no models."""
        with tempfile.TemporaryDirectory() as d:
            A.CONFIG['MLTRAIL_VAULT'] = d
            # an unmounted vault fails loudly
            with self.assertRaises(FileNotFoundError):
                A._registry()


if __name__ == '__main__':
    unittest.main()
