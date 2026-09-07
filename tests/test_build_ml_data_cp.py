"""Unit tests for DATA.build_ML_data_CP (the chemprop analogue of build_ML_data_<ep>).

Synthetic data only — public SMILES (RDKit examples), fake compound ids, random DS_ features. Asserts
the two-pool assembly (cp_pub public-only, cp_int internal-and-measured), the grouping task columns,
the leak filter, and BOTH fold sources: RF's exact folds when the pickle exists, an independent
InChIKey-grouped split when it does not.
"""
import os
import pickle
import tempfile
import types

import numpy as np
import pandas as pd

from python.ADME_build_ML import DATA

# public SMILES only (RDKit examples) — safe stand-ins for real structures
CCO, BENZ, ASP, CAF, TOL = 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'Cc1ccccc1'
NAPH = 'c1ccc2ccccc2c1'   # public, and NOT one of the internal molecules


def _fake_params(rf_dir):
    """Minimal PARAMS stand-in: a sol_lipo grouping over [solubility, logd] and an rf_metrics_dir."""
    return types.SimpleNamespace(
        BEST_CHEMPROP_GROUPINGS={'logd': {'grouping': 'sol_lipo', 'kind': 'endpoint', 'tasks': ['solubility', 'logd']},
                                 'mdck': {'grouping': 'mdck_perm', 'kind': 'cluster', 'target': 'mdck'}},
        CHEMPROP_TRANSFER={'groupings': {'sol_lipo': ['solubility', 'logd']}, 'rf_metrics_dir': rf_dir,
                           'clusters': {'mdck_perm': {'target': 'mdck', 'file': 'tf_fake_cluster.parquet'}}},
        ADME_CACHE=rf_dir, FOLD_GROUP_BY_INCHIKEY=True)


def _data_with_wide():
    """DATA holding a synthetic WIDE df_all (5 internal + 3 public) and an MF_features['all'] with DS_ cols."""
    data = DATA()
    rows = [
        {'compound': 'I_1', 'smiles': CCO,  'source': 'internal', 'origin': 'internal', 'solubility': 1.0, 'logd': 2.0},
        {'compound': 'I_2', 'smiles': BENZ, 'source': 'internal', 'origin': 'internal', 'solubility': np.nan, 'logd': 2.5},
        {'compound': 'I_3', 'smiles': ASP,  'source': 'internal', 'origin': 'internal', 'solubility': 1.5, 'logd': 3.0},
        {'compound': 'I_4', 'smiles': CAF,  'source': 'internal', 'origin': 'internal', 'solubility': -np.inf, 'logd': 1.0},
        {'compound': 'I_5', 'smiles': TOL,  'source': 'internal', 'origin': 'internal', 'solubility': 2.2, 'logd': np.nan},
        {'compound': 'P_1', 'smiles': CCO,  'source': 'EXP', 'origin': 'pub_a', 'solubility': 1.1, 'logd': np.nan},
        {'compound': 'P_2', 'smiles': NAPH, 'source': 'NVS', 'origin': 'nvs',   'solubility': np.nan, 'logd': 2.4},
        {'compound': 'P_3', 'smiles': CAF,  'source': 'ADM', 'origin': 'adm',   'solubility': np.nan, 'logd': np.nan},
    ]
    data.df_all = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    comps = data.df_all.compound.tolist()
    data.MF_features['all'] = pd.DataFrame({'compound': comps, 'DS_a': rng.random(len(comps)),
                                            'DS_b': rng.random(len(comps)), 'MF_0': rng.random(len(comps))})
    return data


def _write_rf_pkl(rf_dir, ep, compounds, folds):
    """Write a minimal <ep>.pkl holding an internal_cv_preddf with compound + fold columns."""
    os.makedirs(rf_dir, exist_ok=True)
    with open(os.path.join(rf_dir, f'{ep}.pkl'), 'wb') as fh:
        pickle.dump({ep: {'internal_cv_preddf': pd.DataFrame({'compound': compounds, 'fold': folds})}}, fh)


def test_pools_tasks_and_features():
    """cp_pub is public-only, cp_int is internal-and-measured, and only DS_ features come along.

    Rows: I_1..I_5 internal (I_5 has no logd) · P_1/P_2 public with a task · P_3 public with none.
    Expect cp_int = 4 internal rows measured for logd, cp_pub = 2 public rows, MF_0 excluded.
    """
    with tempfile.TemporaryDirectory() as td:
        data = _data_with_wide()
        data.build_ML_data_CP(_fake_params(td), k='logd', leak='none', n_splits=2)

        # cp_int keeps only internal rows that measure logd (I_5 has NaN logd)
        assert list(data.cp_int.compound) == ['I_1', 'I_2', 'I_3', 'I_4']
        # cp_pub keeps only public rows with >=1 grouping task measured (P_3 has none)
        assert list(data.cp_pub.compound) == ['P_1', 'P_2']
        # the grouping's task columns come from CHEMPROP_TRANSFER.groupings
        assert data.cp_tasks == ['solubility', 'logd'] and data.cp_grouping == 'sol_lipo' and data.cp_k == 'logd'
        # only the DS_ descriptor columns are carried; the MF_ fingerprint columns are not
        assert data.cp_ds_cols == ['DS_a', 'DS_b'] and 'MF_0' not in data.cp_int.columns


def test_leak_filter_drops_internal_twins():
    """A public row that is the same molecule as an internal compound must leave cp_pub.

    P_1 is ethanol, the same molecule as internal I_1. leak='exact' must drop it; leak='none' must keep it.
    P_2 is naphthalene, which no internal compound matches, so it must survive both levels.
    """
    with tempfile.TemporaryDirectory() as td:
        data = _data_with_wide()
        data.build_ML_data_CP(_fake_params(td), k='logd', leak='exact', n_splits=2)
        # the ethanol twin of I_1 is gone, the naphthalene public row stays
        assert list(data.cp_pub.compound) == ['P_2']

        data.build_ML_data_CP(_fake_params(td), k='logd', leak='none', n_splits=2)
        # with the filter off both public rows survive
        assert list(data.cp_pub.compound) == ['P_1', 'P_2']


def test_folds_from_rf_pickle():
    """When <rf_metrics_dir>/<k>.pkl exists, cp_folds must be RF's exact folds and the source must say so."""
    with tempfile.TemporaryDirectory() as td:
        _write_rf_pkl(td, 'logd', ['I_1', 'I_2', 'I_3', 'I_4'], [0, 1, 0, 1])
        data = _data_with_wide()
        data.build_ML_data_CP(_fake_params(td), k='logd', leak='none')

        # two folds, and each test set is exactly the RF fold membership
        assert len(data.cp_folds) == 2
        assert sorted(data.cp_folds[0][1]) == ['I_1', 'I_3'] and sorted(data.cp_folds[1][1]) == ['I_2', 'I_4']
        # the source string names the pickle it read
        assert data.cp_fold_source == f'rf:{os.path.join(td, "logd.pkl")}'


def test_folds_independent_when_no_pickle():
    """With no RF pickle, cp_folds must be an independent grouped split that still partitions cp_int once."""
    with tempfile.TemporaryDirectory() as td:
        data = _data_with_wide()
        data.build_ML_data_CP(_fake_params(td), k='logd', leak='none', n_splits=2, seed=1)

        # the split is flagged as independent, not RF's
        assert data.cp_fold_source == 'independent'
        # every cp_int compound appears in exactly one test fold
        test_ids = [c for _, te in data.cp_folds for c in te]
        assert sorted(test_ids) == sorted(data.cp_int.compound)
        # train and test never share a compound within a fold
        assert all(not set(tr) & set(te) for tr, te in data.cp_folds)


def test_cluster_grouping_joins_the_novartis_aux_tasks():
    """A cluster grouping must take its auxiliary tasks from the cluster parquet, joined on SMILES.

    The fake parquet gives 2 aux columns for the public molecules only. Expect cp_tasks = [mdck] + the 2 aux,
    aux VALUES on the public rows that match by SMILES, and aux NaN on every internal row (the multitask loss
    masks those), so a cluster endpoint no longer needs run_chemprop_transfer.py.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data = _data_with_wide()
        # internal + public mdck labels, so the target task exists on both sides
        data.df_all['mdck'] = [1.0, 1.2, np.nan, 1.4, 1.5, 0.9, 1.1, np.nan]
        # the cluster parquet carries the aux tasks for two PUBLIC molecules (ethanol, naphthalene)
        pd.DataFrame({'smiles': [CCO, NAPH], '_ik': ['A' * 27, 'B' * 27],
                      'LE_MDCKv2_LogPapp': [0.5, 0.7], 'Caco_2_LogPapp': [0.4, 0.6]}
                     ).to_parquet(os.path.join(tmp, 'tf_fake_cluster.parquet'))
        data.build_ML_data_CP(_fake_params(tmp), k='mdck', leak='none', n_splits=2)

        # the target comes first, then the parquet's aux columns; _ik is never a task
        assert data.cp_tasks == ['mdck', 'LE_MDCKv2_LogPapp', 'Caco_2_LogPapp'], data.cp_tasks
        assert data.cp_grouping == 'mdck_perm'
        # the public ethanol row picked up its aux values through the SMILES join
        p1 = data.cp_pub[data.cp_pub.compound == 'P_1']
        assert len(p1) == 1 and p1['LE_MDCKv2_LogPapp'].iloc[0] == 0.5
        # no internal row has an aux value: internal molecules are not in the Novartis frame
        assert data.cp_int[['LE_MDCKv2_LogPapp', 'Caco_2_LogPapp']].isna().all().all()
        # cp_int still holds every internal row measured for mdck
        assert sorted(data.cp_int.compound) == ['I_1', 'I_2', 'I_4', 'I_5']
