"""Shared helpers for the ADME notebook and port scripts."""
import os
from functools import lru_cache
from glob import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm

IKCACHE = 'autoresearch/predict_adme/_ikcache'   # compound -> _ik parquets, one per public/internal source


def smiles_to_inchikeys(smiles, n_jobs=32, chunk=2000, v=True):
    """Parallel SMILES -> InChIKey with a progress bar.

    Batches the RDKit parse + InChIKey over n_jobs processes and shows one tqdm bar over completed
    batches. Much faster than a per-row comprehension for large sets (e.g. the ~330k MDCK frame).

    :param iterable smiles: SMILES strings
    :param int n_jobs: worker processes
    :param int chunk: SMILES per batch (one tqdm tick per batch)
    :param bool v: show the progress bar
    :return list: InChIKeys aligned to `smiles`; None for an unparseable/non-string SMILES
    """
    smiles = list(smiles)

    def _batch(batch):
        out = []
        for s in batch:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            out.append(Chem.MolToInchiKey(m) if m else None)
        return out

    # split into batches, run in parallel, keep submission order (return_as='generator')
    batches = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    gen = Parallel(n_jobs=n_jobs, return_as='generator')(delayed(_batch)(b) for b in batches)
    res = list(tqdm(gen, total=len(batches), disable=not v, desc='InChIKey'))
    return [ik for r in res for ik in r]


@lru_cache(maxsize=8)
def _load_ikcache(ikcache, id_col):
    """Read every <source>.parquet in the cache directory into one compound -> InChIKey dict (cached)."""
    ikmap = {}
    for f in sorted(glob(os.path.join(ikcache, '*.parquet'))):
        c = pd.read_parquet(f, columns=[id_col, '_ik'])
        ikmap.update(zip(c[id_col], c['_ik']))
    return ikmap


def inchikeys_for(df, id_col='compound', smiles_col='smiles', level='exact', ikcache=IKCACHE, v=False):
    """InChIKeys for a frame, from the `_ikcache` parquets first and RDKit only for what they miss.

    :param dataframe df: must contain id_col and smiles_col
    :param str level: 'exact' = the full 27-char key; 'skeleton' = block 1 only (14 chars), which also
        matches salts, stereoisomers, tautomers and charge states of the same connectivity
    :param str ikcache: directory of <source>.parquet files with columns [compound, _ik]
    :return series: keys aligned to df.index; an unparseable SMILES stays NaN
    """
    # one compound -> _ik lookup across every cached source (built once per directory)
    ikmap = _load_ikcache(ikcache, id_col)
    ik = pd.Series([ikmap.get(c) for c in df[id_col]], index=df.index, dtype=object)
    # RDKit fallback for the compounds the cache does not carry
    if ik.isna().any():
        ik.loc[ik.isna()] = smiles_to_inchikeys(df.loc[ik.isna(), smiles_col], v=v)
    return ik.str[:14] if level == 'skeleton' else ik


def drop_internal_twins(pub, internal, level='exact', id_col='compound', smiles_col='smiles',
                        ikcache=IKCACHE, v=True):
    """Remove every public row that is the SAME MOLECULE as an internal compound (leak control).

    A public row whose InChIKey matches an internal one must not pretrain a transfer model: the frozen
    encoder AND the loaded FFN head can carry a memorized molecule->value mapping into the finetune, even
    though no internal label ever enters the pretrain. Mirrors what `DATA.get_internal_public_sets` does
    for the RF augmented arms. `level='none'` disables the filter and returns `pub` unchanged.

    :param dataframe pub: public frame (the pretrain pool)
    :param dataframe internal: internal frame (every compound that can appear in a validation fold)
    :param str level: 'exact' | 'skeleton' | 'none'; see `inchikeys_for`
    :param bool v: print the dropped count
    :return tuple: (pub without the twins, boolean mask of the dropped rows on the ORIGINAL pub index)
    """
    if level == 'none':
        return pub, pd.Series(False, index=pub.index)
    # match public against internal on the requested key, then drop the matches
    keys_int = set(inchikeys_for(internal, id_col, smiles_col, level, ikcache).dropna())
    leak = inchikeys_for(pub, id_col, smiles_col, level, ikcache).isin(keys_int)
    if v:
        print(f'> leak control ({level}): dropped {int(leak.sum())} of {len(pub)} public rows '
              f'sharing a molecule with the {len(internal)} internal compounds')
    return pub[~leak].reset_index(drop=True), leak


# raw assay units -> modelling space, mirroring the transforms in ADME_build_ML._TF. Defined here
# rather than imported, because ADME_build_ML imports THIS module (an import back would be circular).
_ROUND_TRIP_TF = {'log10':     lambda v: np.log10(v),
                  'logit_pct': lambda v: np.log10(v / (100.0 - v)),
                  'identity':  lambda v: v}


def score_round_trip(preds, truth, endpoints, id_col='id', truth_id='name',
                     pred_col=lambda k: k + '_pred', conf_col=lambda k: k + '_confidence', v=True):
    """
    -Score ANY table of raw-unit predictions against the internal experimental truth, endpoint by
     endpoint, applying exactly the steps the training path applies so that two prediction sets stay
     comparable: the config `filter` (mdck keeps MDR1 only), `label_cap_raw` winsorization, the
     non-finite drop, and the config transform into MODELLING space. Used for the deployed-model
     round trip AND for scoring an external model (which has no confidence column).
    param dataframe preds: one row per compound; id_col joins to truth[truth_id]
    param dataframe truth: the internal experimental pull (DATA.df_internal_exp_all)
    param dict endpoints: params.ADME_ENDPOINTS (col / transform / unit / filter / label_cap_raw)
    param callable pred_col: endpoint key -> its prediction column in `preds`
    param callable conf_col: endpoint key -> its confidence column, or None when there is none
    param bool v: print how many compounds matched
    return tuple: (long [compound, endpoint, real_y, pred_y, conf, resid], metrics indexed by endpoint)
    """
    import ML_Reg                                    # lazy: only the notebook/runner path needs it
    rt = truth.merge(preds, left_on=truth_id, right_on=id_col, how='inner')
    if v:
        print(f'> round trip: {len(rt)} of {len(truth)} internal compounds matched a prediction')
    cols = {'R2_det': 'r2det', 'R2_pears': 'r2', 'RMSE': 'rmse', 'N': 'n_test'}
    long, metrics = [], []
    for k, ep in endpoints.items():
        d = rt
        # mdck keeps only the MDR1 cell line, exactly as DATA._endpoint_dfs does
        if ep.get('filter'):
            d = d[d[ep['filter']['col']].astype(str).str.contains(ep['filter']['contains'], na=False)]
        tf = _ROUND_TRIP_TF[ep['transform']]
        # winsorize the truth at the training cap, so a saturated assay value is not scored as an error
        real = tf(d[ep['col']].astype(float).clip(upper=ep.get('label_cap_raw')))
        pred = tf(d[pred_col(k)].astype(float))
        m = np.isfinite(real) & np.isfinite(pred)
        cc = conf_col(k) if conf_col else None
        long.append(pd.DataFrame({'compound': d.loc[m, truth_id], 'endpoint': k,
                                  'real_y': real[m], 'pred_y': pred[m],
                                  'conf': d.loc[m, cc].astype(float) if cc else np.nan}))
        r = ML_Reg.get_reg_metrics_from_preddf(long[-1], v=False)
        metrics.append({'endpoint': k, 'unit': ep.get('unit', ''),
                        # bias = mean(pred - real): a non-zero value is exactly what R2_pears hides
                        'bias': float(np.mean(long[-1].pred_y - long[-1].real_y)),
                        **{lab: r[key] for lab, key in cols.items()}})
    long = pd.concat(long, ignore_index=True)
    long['resid'] = long['pred_y'] - long['real_y']
    return long, pd.DataFrame(metrics).set_index('endpoint')


# -------------------------------
# UNHASHED Morgan fingerprints — MOVED to Scripts/Rdkit_tools.py on 2026-09-15
# -------------------------------
# unhashed_morgan / morgan_bit_info / draw_bit_on_molecule / plot_decision_tree_MF_bits and the
# BIT_COLORS palette now live next to their hashed siblings (get_MF_bits_from_df,
# compute_H236_features, compute_H237_features) in Scripts/Rdkit_tools.py, so all the chemistry
# and depiction code sits in one module. Call them as rdkit_tools.<name>.
