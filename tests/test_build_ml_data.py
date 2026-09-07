"""Unit tests for the generalized DATA.build_ML_data_<ep> (generic DATA._endpoint_ML).

Synthetic data only — public SMILES (RDKit examples), fake compound ids, random features. No real
chemistry, nothing crosses the wire. Asserts the generic modelling-frame assembly: the config label
cap (label_cap_raw), the feature merge, the NaN/inf-label drop, and the SMILES dedup (internal wins).
"""
import tempfile
import types

import numpy as np
import pandas as pd

from python.ADME_build_ML import DATA

# public SMILES only (RDKit examples) — safe stand-ins for real structures
CCO, BENZ, ASP, CAF = 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'


def _fake_params():
    """Minimal PARAMS stand-in: solubility (log10 + 15000 µM cap) and logd (identity, no cap)."""
    return types.SimpleNamespace(ADME_ENDPOINTS={
        'solubility': {'col': 'x', 'transform': 'log10', 'unit': 'uM', 'label_cap_raw': 15000},
        'logd':       {'col': 'x', 'transform': 'identity', 'unit': 'logD7.4'},
    })


def _data_with(endpoint, dfs_rows, feat_compounds):
    """Build a DATA with a synthetic dfs[endpoint] and an MF_features['all'] over feat_compounds."""
    data = DATA()
    data.params = _fake_params()
    data.dfs[endpoint] = pd.DataFrame(dfs_rows)
    rng = np.random.default_rng(0)
    data.MF_features['all'] = pd.DataFrame({'compound': list(feat_compounds),
                                            'F0': rng.random(len(feat_compounds)),
                                            'F1': rng.random(len(feat_compounds))})
    return data


def test_solubility_cap_merge_dedup_and_finite():
    """Solubility frame: cap at log10(15000), inner-merge on features, drop -inf label, dedup smiles (internal wins).

    Rows: C1 internal CCO @20000µM (above cap) · C2 EXP benzene @500 · C3 EXP CCO @300 (smiles-twin of C1)
    · C4 EXP aspirin @0µM (log10 -> -inf) · C5 EXP caffeine but NOT featurized. Expect only C1 (capped,
    internal-preferred over C3) and C2 to survive.
    """
    rows = [
        {'compound': 'C1', 'smiles': CCO,  'label': np.log10(20000.0), 'source': 'internal', 'origin': 'internal'},
        {'compound': 'C2', 'smiles': BENZ, 'label': np.log10(500.0),   'source': 'EXP',      'origin': 'pub_a'},
        {'compound': 'C3', 'smiles': CCO,  'label': np.log10(300.0),   'source': 'EXP',      'origin': 'pub_a'},
        {'compound': 'C4', 'smiles': ASP,  'label': -np.inf,           'source': 'EXP',      'origin': 'pub_b'},
        {'compound': 'C5', 'smiles': CAF,  'label': np.log10(800.0),   'source': 'EXP',      'origin': 'pub_b'},
    ]
    data = _data_with('solubility', rows, feat_compounds=['C1', 'C2', 'C3', 'C4'])   # C5 absent -> merge-dropped
    data.build_ML_data_solubility()
    ml = data.ML_data['solubility']

    # exactly two molecules survive (dup smiles collapsed, -inf dropped, unfeaturized dropped)
    assert len(ml) == 2
    # unique smiles are CCO + benzene
    assert set(ml['smiles']) == {CCO, BENZ}
    # the CCO row is the INTERNAL one (dedup prefers internal over the EXP twin C3)
    assert ml.loc[ml['smiles'] == CCO, 'source'].iloc[0] == 'internal'
    assert ml.loc[ml['smiles'] == CCO, 'compound'].iloc[0] == 'C1'
    # label capped: the CCO label equals log10(15000), not log10(20000)
    assert np.isclose(ml.loc[ml['smiles'] == CCO, 'label'].iloc[0], np.log10(15000.0))
    # every label is finite
    assert np.isfinite(ml['label']).all()
    # meta + feature columns present
    assert {'compound', 'smiles', 'label', 'source', 'origin', 'F0', 'F1'} <= set(ml.columns)


def test_logd_has_no_cap():
    """LogD has no label_cap_raw, so a large label passes through unchanged (identity transform)."""
    rows = [{'compound': 'C1', 'smiles': CCO, 'label': 20000.0, 'source': 'internal', 'origin': 'internal'}]
    data = _data_with('logd', rows, feat_compounds=['C1'])
    data.build_ML_data_logd()
    # the label is NOT capped (no label_cap_raw for logd)
    assert data.ML_data['logd']['label'].iloc[0] == 20000.0


def test_all_eight_aliases_exist():
    """Every endpoint exposes generic get_<ep>_data and build_ML_data_<ep> (route through the _endpoint_* builders)."""
    for ep in ('solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb'):
        # both aliases resolve to callables on DATA
        assert callable(getattr(DATA, f'get_{ep}_data'))
        assert callable(getattr(DATA, f'build_ML_data_{ep}'))


def test_endpoint_dfs_transform_filter_concat_and_raw():
    """Generic _endpoint_dfs: internal transform, NaN-drop, cell-line filter, multi-file public concat, raw=inverse.

    Synthetic internal frame + two fake public parquets in a temp cache. 'sol' (log10, one public file);
    'mdck' (log10 + MDR1 filter, two public files concatenated). Public SMILES only, fake ids.
    """
    internal = pd.DataFrame({
        'name': ['I1', 'I2', 'I3'],
        'smiles': [CCO, BENZ, ASP],
        'sol_col': [100.0, 1000.0, np.nan],            # I3 NaN -> dropped for sol
        'mdck_col': [10.0, 5.0, 2.0],
        'mdck_Cell line': ['MDR1', 'MDR2', 'MDR1'],    # I2 filtered out for mdck
    })
    with tempfile.TemporaryDirectory() as cache:
        # two public parquets (value already in modelling space) to exercise multi-file concat
        pd.DataFrame({'compound': ['P1'], 'smiles': [CAF], 'value': [np.log10(50.0)], 'origin': ['nvs']}).to_parquet(f'{cache}/pub1.parquet')
        pd.DataFrame({'compound': ['P2'], 'smiles': [BENZ], 'value': [np.log10(20.0)], 'origin': ['adm']}).to_parquet(f'{cache}/pub2.parquet')
        params = types.SimpleNamespace(
            ADME_CACHE=cache,
            ADME_ENDPOINTS={
                'sol':  {'col': 'sol_col', 'transform': 'log10'},
                'mdck': {'col': 'mdck_col', 'transform': 'log10', 'filter': {'col': 'mdck_Cell line', 'contains': 'MDR1'}},
            },
            ENDPOINT_PUBLIC_FILES={'sol': ['pub1.parquet'], 'mdck': ['pub1.parquet', 'pub2.parquet']},
        )
        data = DATA()
        data.params = params
        data.df_internal_exp_all = internal

        # sol: I3 dropped (NaN), I1/I2 internal + P1 public = 3 rows
        data._endpoint_dfs(params, 'sol')
        s = data.dfs['sol']
        assert len(s) == 3
        assert set(s[s.source == 'internal']['compound']) == {'I1', 'I2'}
        # internal label log10-transformed and raw = 10**label
        assert np.isclose(s.loc[s.compound == 'I1', 'label'].iloc[0], np.log10(100.0))
        assert np.isclose(s.loc[s.compound == 'I1', 'raw'].iloc[0], 100.0)

        # mdck: MDR1 filter keeps I1/I3 (drops I2=MDR2); both public files concatenated
        data._endpoint_dfs(params, 'mdck')
        md = data.dfs['mdck']
        assert set(md[md.source == 'internal']['compound']) == {'I1', 'I3'}
        assert set(md[md.source == 'EXP']['compound']) == {'P1', 'P2'}


def test_grouped_folds_no_inchikey_twins_across_folds():
    """DATA._grouped_folds keeps InChIKey twins in one fold, tests each compound once, and is deterministic.

    Input: 40 internal compounds paired into 20 InChIKeys (twins), plus one twin pair with a missing key.
    Expected: folds partition the set; no molecule (shared _ik) straddles train/test; same seed -> same folds.
    Rationale: guards the anti-leakage grouped CV (config FOLD_GROUP_BY_INCHIKEY).
    """
    n = 40
    comp = [f'SRB-{i:06d}-001' for i in range(n)]
    ik = [f'IK{i // 2}' for i in range(n)]          # each InChIKey shared by 2 consecutive compounds (twins)
    ik[0] = ik[1] = None                            # a twin pair with a missing key -> singleton groups
    d = DATA()
    d.internal = pd.DataFrame({'compound': comp, '_ik': ik})
    folds = d._grouped_folds(n_splits=5, seed=42)

    # every internal compound is tested exactly once (folds partition the set)
    assert sorted(c for _, te in folds for c in te) == sorted(comp)
    ikmap = dict(zip(comp, ik))
    for tr, te in folds:
        # no molecule's InChIKey appears in both train and test of a fold
        te_iks = {ikmap[c] for c in te if ikmap[c] is not None}
        tr_iks = {ikmap[c] for c in tr if ikmap[c] is not None}
        assert te_iks.isdisjoint(tr_iks)
    # same seed reproduces identical folds
    assert d._grouped_folds(5, 42) == folds


def test_build_ML_data_RF_matches_the_alias():
    """build_ML_data_RF(params, k) must produce the same frame as the build_ML_data_<ep> alias.

    Two DATA objects get the identical synthetic logd rows. One runs the alias, the other the named entry
    point. Both frames must be equal, and the named one must not need self.params set in advance.
    """
    rows = [
        {'compound': 'C1', 'smiles': CCO,  'label': 2.0, 'source': 'internal', 'origin': 'internal'},
        {'compound': 'C2', 'smiles': BENZ, 'label': 3.1, 'source': 'EXP',      'origin': 'pub_a'},
    ]
    alias = _data_with('logd', rows, feat_compounds=['C1', 'C2'])
    alias.build_ML_data_logd()
    named = _data_with('logd', rows, feat_compounds=['C1', 'C2'])
    named.params = None                                        # the named entry point binds params itself
    named.build_ML_data_RF(_fake_params(), k='logd')

    # the two frames are identical
    pd.testing.assert_frame_equal(alias.ML_data['logd'], named.ML_data['logd'])
    # the named entry point stored the params it was given
    assert named.params is not None
