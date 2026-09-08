"""Unit tests for OUTPUT.assess_predictions_cp (the 3 chemprop arms) and the ml_model dispatch.

No real chemprop runs: a STUB `chemprop` script is written per test run. It records its argv, fabricates a
checkpoint on `train`, and fabricates a prediction column on `predict`. Synthetic frames only — public
SMILES, fake compound ids. This checks the plumbing (arm semantics, provenance, pickle round trip), not
model quality.
"""
import os
import stat
import sys
import tempfile
import types

import numpy as np
import pandas as pd

from python.ADME_build_ML import DATA, OUTPUT
import tests.test_build_ml_data_cp as cp

RF_CFG = {'champion': {'n_estimators': 5}, 'seed': 42, 'n_jobs': 1}

STUB = """#!/usr/bin/env bash
# stub chemprop: record argv, fabricate a checkpoint on train, fabricate preds on predict
echo "$@" >> "$FAKE_CP_LOG"
if [ "$1" = train ]; then
  out=""; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out=$2; shift; done
  mkdir -p "$out/model_0"; touch "$out/model_0/best.pt"; touch "$out/best.ckpt"
else
  inp=""; pp=""; while [ $# -gt 0 ]; do [ "$1" = "-i" ] && inp=$2; [ "$1" = "--preds-path" ] && pp=$2; shift; done
  n=$(( $(wc -l < "$inp") - 1 ))
  { echo "smiles,pred_0,pred_1"; for i in $(seq 1 $n); do echo "CCO,1.$i,9.9"; done; } > "$pp"
fi
"""


def _stub_bin(tmp):
    """Write the stub chemprop script into tmp, point FAKE_CP_LOG at a fresh log, and return (bin, log)."""
    path, log = os.path.join(tmp, 'chemprop'), os.path.join(tmp, 'argv.log')
    open(path, 'w').write(STUB)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    open(log, 'w').close()
    os.environ['FAKE_CP_LOG'] = log
    return path, log


def _output(tmp):
    """OUTPUT with a stub RF config, so no real forest is built."""
    return OUTPUT(types.SimpleNamespace(RF_SINGLETASK=RF_CFG))


def _cp_params(tmp):
    """Fake PARAMS whose CHEMPROP_TRANSFER points at the stub binary and at scratch dirs under tmp."""
    prm = cp._fake_params(tmp)
    prm.CHEMPROP_TRANSFER.update({
        'chemprop_bin': _stub_bin(tmp)[0], 'output_dir': os.path.join(tmp, 'runs'),
        'pretrain_dir': os.path.join(tmp, 'pretrain'), 'epochs_pretrain': 2, 'epochs_finetune': 2,
        'patience': 1, 'freeze_encoder': True, 'descriptors': 'precomputed',
        'molecule_featurizers': ['v1_rdkit_2d_normalized'], 'hp': {'depth': 3}})
    return prm


def _data(tmp):
    """DATA with cp_* built from the synthetic wide frame."""
    d = cp._data_with_wide()
    d.build_ML_data_CP(_cp_params(tmp), k='logd', leak='none', n_splits=2)
    return d


def test_three_arms_recorded_with_provenance():
    """assess_predictions_cp must record augmented_cv, internal_cv and ext_->_internal, plus provenance."""
    with tempfile.TemporaryDirectory() as tmp:
        data, out = _data(tmp), _output(tmp)
        r = out.assess_predictions_cp(data, os.path.join(tmp, 'pkl', 'logd.pkl'))

        # the three RF-equivalent arms each have a preddf and a metrics dict
        for arm in ('augmented_cv', 'internal_cv', 'ext_->_internal'):
            assert arm + '_preddf' in r and arm + '_metrics' in r, arm
            assert set(['compound', 'real_y', 'pred_y', 'residuals']) <= set(r[arm + '_preddf'].columns)
        # the CV arms cover every cp_int compound once; ext_->_internal predicts all of them in one block
        assert len(r['augmented_cv_preddf']) == len(data.cp_int) == len(r['ext_->_internal_preddf'])
        # provenance travels with the result
        assert r['_grouping'] == 'sol_lipo' and r['_tasks'] == ['solubility', 'logd']
        assert r['_fold_source'] == 'independent' and r['_pretrain_checkpoint'].endswith('.ckpt')
        # the store is keyed by endpoint
        assert list(out.metrics_results_cp) == ['logd']


def test_arm_semantics_from_the_argv_log():
    """The stub argv must show: one pretrain, folds with --checkpoint, scratch folds without, one bare predict."""
    with tempfile.TemporaryDirectory() as tmp:
        data, out = _data(tmp), _output(tmp)
        out.assess_predictions_cp(data, os.path.join(tmp, 'pkl', 'logd.pkl'))
        argv = open(os.environ['FAKE_CP_LOG']).read().splitlines()

        trains = [a for a in argv if a.startswith('train')]
        # 1 pretrain + 2 transfer folds + 2 scratch folds
        assert len(trains) == 5, trains
        # exactly the 2 transfer folds load the checkpoint and freeze the encoder
        assert sum('--checkpoint' in a for a in trains) == 2
        assert sum('--freeze-encoder' in a for a in trains) == 2
        # ext_->_internal is a predict with no training of its own: 2+2 fold predicts + 1 = 5
        assert len([a for a in argv if a.startswith('predict')]) == 5, argv


def test_pickle_round_trip_and_dispatch():
    """A second call loads the pickle instead of retraining, and ml_model='chemprop' reaches this method."""
    with tempfile.TemporaryDirectory() as tmp:
        data, out = _data(tmp), _output(tmp)
        path = os.path.join(tmp, 'pkl', 'logd.pkl')
        first = out.assess_predictions_cp(data, path)

        open(os.environ['FAKE_CP_LOG'], 'w').close()
        out2 = _output(tmp)
        # the dispatcher must route to the chemprop body and load from disk
        second = out2.assess_predictions(data, path, ml_model='chemprop')
        assert open(os.environ['FAKE_CP_LOG']).read().strip() == ''            # nothing ran; it came from the pickle
        assert second['_grouping'] == first['_grouping']
        assert np.allclose(second['augmented_cv_preddf'].pred_y, first['augmented_cv_preddf'].pred_y)


def test_assess_all_endpoints_cp_loops_and_collects():
    """assess_all_endpoints_cp must build each endpoint's pools, write one pickle each, and collect them.

    Two endpoints share the sol_lipo grouping here, so the stage-1 pretrain must run ONCE and be reused by
    the second endpoint (the argv log shows one pretrain, not two).
    """
    with tempfile.TemporaryDirectory() as tmp:
        data = cp._data_with_wide()
        prm = _cp_params(tmp)
        prm.BEST_CHEMPROP_GROUPINGS['solubility'] = 'sol_lipo'
        prm.METRICS_PKL_CP_DIR = os.path.join(tmp, 'cp_pkl')
        data.build_ML_data_CP(prm, k='logd', leak='none', n_splits=2)     # binds data.params
        out = _output(tmp)
        r = out.assess_all_endpoints_cp(data, prm, endpoints=['logd', 'solubility'], leak='none', n_splits=2)

        # both endpoints are collected, keyed by endpoint
        assert sorted(r) == ['logd', 'solubility'] == sorted(out.metrics_results_cp)
        # one pickle per endpoint
        assert sorted(os.listdir(os.path.join(tmp, 'cp_pkl'))) == ['logd.pkl', 'solubility.pkl']
        # the shared sol_lipo pretrain ran once, so only 1 of the train calls has no --checkpoint
        trains = [a for a in open(os.environ['FAKE_CP_LOG']).read().splitlines() if a.startswith('train')]
        assert sum('--checkpoint' not in a for a in trains) == 1 + 4, trains   # 1 pretrain + 4 scratch folds
