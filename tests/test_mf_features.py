"""Unit tests for DATA.build_MF_features — the public cache / internal recompute split.

The real H237 block is replaced by a counting stub, so these tests check the CACHE LOGIC, not the
chemistry: what is computed, what is read from disk, when the public cache is topped up, and that the
coverage assertion fires. Synthetic frames only — public SMILES, fake compound ids.
"""
import os
import tempfile
import types

import numpy as np
import pandas as pd

from python.ADME_build_ML import DATA

# public SMILES only (RDKit examples) — safe stand-ins for real structures
CCO, BENZ, ASP, CAF, TOL = 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'Cc1ccccc1'
FEATS = ['MF_0', 'DS_MolWt']


def _params(tmp):
    """Minimal PARAMS stand-in: only the features directory matters here."""
    return types.SimpleNamespace(MF_features_all_path=tmp)


def _data(rows):
    """DATA holding a synthetic WIDE df_all, with _compute_MF replaced by a counting stub."""
    data = DATA()
    data.df_all = pd.DataFrame(rows)
    calls = []

    def stub(smi, type, n_jobs):
        # record what each call was asked to featurize, then return deterministic fake features
        calls.append(list(smi['compound']))
        return pd.DataFrame({'compound': list(smi['compound']),
                             'MF_0': np.arange(len(smi), dtype='int8'),
                             'DS_MolWt': np.arange(len(smi), dtype=float)})

    data._compute_MF = stub
    return data, calls


ROWS = [
    {'compound': 'I_1', 'smiles': CCO,  'source': 'internal', 'origin': 'internal'},
    {'compound': 'I_2', 'smiles': BENZ, 'source': 'internal', 'origin': 'internal'},
    {'compound': 'P_1', 'smiles': ASP,  'source': 'EXP', 'origin': 'pub_a'},
    {'compound': 'P_2', 'smiles': CAF,  'source': 'NVS', 'origin': 'nvs'},
]


def test_first_call_computes_both_blocks_and_caches_public_only():
    """A cold start must compute the public block AND the internal block, and cache only the public one."""
    with tempfile.TemporaryDirectory() as tmp:
        data, calls = _data(ROWS)
        data.build_MF_features(_params(tmp), type='H237')

        # two compute calls: the public block then the internal block
        assert calls == [['P_1', 'P_2'], ['I_1', 'I_2']], calls
        # the cache holds the PUBLIC compounds only; internal is never written to disk
        cached = pd.read_parquet(os.path.join(tmp, '20260824_MF_features_H237_public.parquet'))
        assert sorted(cached.compound) == ['P_1', 'P_2']
        # the assembled matrix covers every df_all compound
        assert sorted(data.MF_features['all'].compound) == ['I_1', 'I_2', 'P_1', 'P_2']


def test_second_call_reads_the_public_cache_and_recomputes_internal():
    """A warm start must read the public cache from disk and still recompute the internal block."""
    with tempfile.TemporaryDirectory() as tmp:
        data, _ = _data(ROWS)
        data.build_MF_features(_params(tmp), type='H237')

        data2, calls2 = _data(ROWS)
        data2.build_MF_features(_params(tmp), type='H237')
        # exactly one compute call, for the internal block; the public block came from the cache
        assert calls2 == [['I_1', 'I_2']], calls2
        assert len(data2.MF_features['all']) == 4


def test_new_internal_compound_is_picked_up():
    """THE BUG THIS FIXES: a CDD pull adds an internal compound, and it must get features immediately.

    The public cache is written first, then a third internal compound appears in df_all. The old
    all-compound cache would have missed it and _endpoint_ML would drop it silently.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data, _ = _data(ROWS)
        data.build_MF_features(_params(tmp), type='H237')

        grown = ROWS + [{'compound': 'I_3', 'smiles': TOL, 'source': 'internal', 'origin': 'internal'}]
        data2, calls2 = _data(grown)
        data2.build_MF_features(_params(tmp), type='H237')
        # the new internal compound was featurized with the others
        assert calls2 == [['I_1', 'I_2', 'I_3']], calls2
        # and it is present in the assembled matrix, so no row can be dropped downstream
        assert 'I_3' in set(data2.MF_features['all'].compound) and len(data2.MF_features['all']) == 5


def test_new_public_compound_tops_up_the_cache():
    """A new public compound must be computed once and appended to the cache, not trigger a full rebuild."""
    with tempfile.TemporaryDirectory() as tmp:
        data, _ = _data(ROWS)
        data.build_MF_features(_params(tmp), type='H237')

        grown = ROWS + [{'compound': 'P_3', 'smiles': TOL, 'source': 'EXP', 'origin': 'pub_b'}]
        data2, calls2 = _data(grown)
        data2.build_MF_features(_params(tmp), type='H237')
        # only the ONE new public compound is computed, plus the internal block
        assert calls2 == [['P_3'], ['I_1', 'I_2']], calls2
        # the top-up was persisted, so the next run reads all three public compounds
        cached = pd.read_parquet(os.path.join(tmp, '20260824_MF_features_H237_public.parquet'))
        assert sorted(cached.compound) == ['P_1', 'P_2', 'P_3']


def test_legacy_all_compound_cache_is_derived_not_recomputed():
    """An existing <date>_MF_features.parquet must seed the public cache, so no public recompute happens."""
    with tempfile.TemporaryDirectory() as tmp:
        # the legacy cache holds every compound, internal included, exactly as the old builder wrote it
        pd.DataFrame({'compound': ['I_1', 'I_2', 'P_1', 'P_2'], 'MF_0': np.zeros(4, dtype='int8'),
                      'DS_MolWt': np.zeros(4)}).to_parquet(os.path.join(tmp, '20260824_MF_features.parquet'))
        data, calls = _data(ROWS)
        data.build_MF_features(_params(tmp), type='H237')

        # only the internal block is computed; the public block is sliced out of the legacy file
        assert calls == [['I_1', 'I_2']], calls
        # the derived cache keeps the public rows only
        cached = pd.read_parquet(os.path.join(tmp, '20260824_MF_features_H237_public.parquet'))
        assert sorted(cached.compound) == ['P_1', 'P_2']


def test_column_mismatch_between_blocks_raises():
    """A public cache from another feature version must fail loudly, not concatenate into NaN columns."""
    with tempfile.TemporaryDirectory() as tmp:
        # the cached public block carries a feature the fresh internal block does not produce
        pd.DataFrame({'compound': ['P_1', 'P_2'], 'MF_0': np.zeros(2, dtype='int8'),
                      'DS_MolWt': np.zeros(2), 'DS_OLD_ONLY': np.zeros(2)}
                     ).to_parquet(os.path.join(tmp, '20260824_MF_features_H237_public.parquet'))
        data, _ = _data(ROWS)
        try:
            data.build_MF_features(_params(tmp), type='H237')
            raised = False
        except AssertionError as ex:
            raised = 'another feature version' in str(ex)
        assert raised


def test_unknown_feature_type_raises():
    """Only 'H237' and 'H236' are valid feature types."""
    with tempfile.TemporaryDirectory() as tmp:
        data = DATA()
        data.df_all = pd.DataFrame(ROWS)
        try:
            data.build_MF_features(_params(tmp), type='H999')
            raised = False
        except ValueError as ex:
            raised = 'unknown feature type' in str(ex)
        assert raised
