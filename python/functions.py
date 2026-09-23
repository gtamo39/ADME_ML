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


def ordered_endpoints(params):
    """
    -Endpoint keys in the canonical DISPLAY order, so every figure puts the bars/panels in the same
     places. The order is the webapp column order (config webapp.endpoint_order); an endpoint that key
     omits is appended, so a new config endpoint is never silently dropped from a figure.
     Mirrors webapp/app.py::_ordered_endpoints.
    param class params: PARAMS instance (ADME_ENDPOINTS, webapp)
    return list: endpoint keys
    """
    order = params.webapp.get('endpoint_order', [])
    return ([k for k in order if k in params.ADME_ENDPOINTS]
            + [k for k in params.ADME_ENDPOINTS if k not in order])


# the three prediction arms both bar figures show, in draw order: key -> legend label
_BAR_ARMS = {'ext_->_internal': 'Transfer (public→internal)', 'internal_cv': 'Internal CV',
             'augmented_cv': 'Public-augmented CV'}
_METRIC_LABEL = {'r2': 'in-house R² (squared Pearson)',
                 'r2det': 'in-house R²det (coefficient of determination)'}
_METRIC_SHORT = {'r2': 'R²', 'r2det': 'R²det'}


def _load_metrics_pkls(directory, endpoints, in_memory=None):
    """
    -Read <directory>/<endpoint>.pkl for each endpoint, preferring an in-memory store when given.
     A pickle may hold either the endpoint's dict or a {endpoint: dict} wrapper; both are unwrapped.
    param str directory: the metrics-pickle directory
    param list endpoints: endpoint keys to load
    param dict in_memory: results already in memory (e.g. output.metrics_results), preferred
    return tuple: ({endpoint: results}, [endpoints with no pickle])
    """
    import pickle
    out, missing = {}, []
    for ep in endpoints:
        if in_memory and ep in in_memory:
            out[ep] = in_memory[ep]
            continue
        path = os.path.join(directory, f'{ep}.pkl')
        if not os.path.exists(path):
            missing.append(ep)
            continue
        with open(path, 'rb') as fh:
            loaded = pickle.load(fh)
        out[ep] = loaded[ep] if isinstance(loaded, dict) and ep in loaded else loaded
    return out, missing


def _save_fig(fig, out_png, dpi, default_name, v=True):
    """
    -Write the figure only when dpi is given: dpi is the SAVE SWITCH, so None writes nothing.
    param str out_png: destination; None falls back to output/figures/<default_name>
    param int dpi: resolution, and the switch — None means do not write a file
    return str: the path written, or None
    """
    if dpi is None:
        return None
    out_png = out_png or os.path.join('output/figures', default_name)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')   # before show(): some backends blank the figure
    if v:
        print(f'> saved {out_png} (dpi={dpi})')
    return out_png


def plot_bar_summary_RF_all_endpoints(rf_dir, params, metric='r2', rf_results=None,
                                      out_png=None, dpi=None, v=True):
    """
    -Grouped bar summary of the RF arms over every endpoint: transfer (public->internal), internal CV
     and public-augmented CV. One model family only — see plot_bar_summary_RF_CP_all_endpoints for the
     figure that also puts chemprop and near-shell in the contest. Values come from the per-endpoint
     pickles, so nothing is refitted. Endpoints run in the canonical display order (ordered_endpoints);
     one with no pickle is skipped and named.
    param str rf_dir: params.METRICS_PKL_RF_DIR — holds <endpoint>.pkl with the 6 RF arms
    param class params: PARAMS instance (ADME_ENDPOINTS, webapp.endpoint_order, SERAC_C)
    param str metric: 'r2' (squared Pearson, affine-invariant) | 'r2det' (coeff. of determination)
    param dict rf_results: in-memory results (output.metrics_results) preferred over rf_dir
    param str out_png: PNG path; default output/figures/bar_summary_RF_<metric>.png
    param int dpi: resolution AND the save switch — None writes NO file, the figure is only shown
    param bool v: print the numbers behind the bars
    return tuple: (figure, DataFrame indexed by endpoint)
    """
    import matplotlib.pyplot as plt          # lazy: keeps this module importable without a display

    rf, missing = _load_metrics_pkls(rf_dir, ordered_endpoints(params), rf_results)
    if missing and v:
        print(f'! no RF pickle in {rf_dir} for {missing} — excluded from the figure')
    eps = [ep for ep in ordered_endpoints(params) if ep in rf]
    # one row per endpoint: the internal test size plus the chosen metric for each arm
    df = pd.DataFrame([{'n': rf[ep].get('internal_cv_metrics', {}).get('n_test', 0),
                        **{a: rf[ep].get(a + '_metrics', {}).get(metric, np.nan) for a in _BAR_ARMS}}
                       for ep in eps], index=eps)

    # R2det can go strongly negative on an uncalibrated arm -> adaptive limits; R2 keeps the fixed frame
    vals = [v2 for a in _BAR_ARMS for v2 in df[a] if np.isfinite(v2)]
    if metric == 'r2':
        lo, hi, ny = 0.0, 1.12, 1.01
    else:
        pad = 0.14 * (max(vals) - min(vals))
        lo, hi, ny = min(0.0, min(vals)) - 0.4 * pad, max(vals) + 1.6 * pad, max(vals) + 0.35 * pad

    colors = [params.SERAC_C['paper'], params.SERAC_C['ember'], params.SERAC_C['azure']]
    x = np.arange(len(eps)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    # one value-labeled bar group per arm
    for i, (arm, label) in enumerate(_BAR_ARMS.items()):
        bars = ax.bar(x + (i - 1) * w, df[arm], w, label=label, color=colors[i],
                      edgecolor='black', linewidth=0.4)
        ax.bar_label(bars, fmt='%.2f', fontsize=6, padding=1)
    # zero reference line whenever bars go below it
    if lo < 0:
        ax.axhline(0, color='#444', lw=0.8, zorder=1)
    # internal test size on top of each endpoint
    for xi, n in zip(x, df['n']):
        ax.text(xi, ny, f'n={int(n)}', ha='center', va='bottom', fontsize=7, color='#444')
    ax.set_xticks(x); ax.set_xticklabels(eps)
    ax.set_ylabel(_METRIC_LABEL[metric]); ax.set_ylim(lo, hi)
    ax.set_title(f'ADME endpoint prediction — {_METRIC_SHORT[metric]} by RF arm (internal target)', pad=26)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
    _save_fig(fig, out_png, dpi, f'bar_summary_RF_{metric}.png', v)
    if v:
        print(f'RF arms, {metric}')
        print(df.round(3).to_string())
    return fig, df


def plot_bar_summary_RF_CP_all_endpoints(rf_dir, cp_dir, params, metric='r2', clip0=True,
                                         nearshell_dir='output/results/20260831_ADME_nearshell',
                                         cp_results=None, out_png=None, dpi=None, v=True):
    """
    -Grouped bar summary over every endpoint: transfer (public->internal), internal CV and
     public-augmented CV. Each bar is the WINNER of RF vs chemprop (vs near-shell, augmented arm only),
     so the figure answers "how well can we predict this endpoint", not "how well does one model do".
     Values come from the per-endpoint pickles, so nothing is refitted. Endpoints run in the canonical
     display order (ordered_endpoints), and one with no RF pickle is skipped and named.
    param str rf_dir: params.METRICS_PKL_RF_DIR — holds <endpoint>.pkl with the 6 RF arms
    param str cp_dir: params.METRICS_PKL_CP_DIR — holds <endpoint>.pkl with the 3 chemprop arms
    param class params: PARAMS instance (ADME_ENDPOINTS, webapp.endpoint_order, SERAC_C)
    param str metric: 'r2' (squared Pearson, affine-invariant) | 'r2det' (coeff. of determination)
    param bool clip0: floor negative r2det bars at 0 for readability; labels keep the TRUE value
    param str nearshell_dir: directory of nested_rf_<endpoint>.json; None or absent -> arm skipped
    param dict cp_results: in-memory chemprop results (output.metrics_results_cp) preferred over cp_dir
    param str out_png: PNG path; default output/figures/bar_summary_RF_CP_<metric>.png
    param int dpi: resolution AND the save switch — None writes NO file, the figure is only shown
    param bool v: print the per-endpoint winner table
    return tuple: (figure, summary DataFrame indexed by endpoint)
    """
    import json
    import matplotlib.pyplot as plt          # lazy: keeps this module importable without a display

    LBL = {'RF': 'RF', 'cp': 'chemprop', 'ns': 'near-shell'}
    ns_key = {'r2': 'honest_r2', 'r2det': 'honest_r2det'}[metric]

    # RF is the only mandatory source: an endpoint without its pickle has no bar to draw
    rf, missing = _load_metrics_pkls(rf_dir, ordered_endpoints(params))
    if missing and v:
        print(f'! no RF pickle in {rf_dir} for {missing} — excluded from the figure')
    eps = [ep for ep in ordered_endpoints(params) if ep in rf]
    # chemprop: the in-memory store wins, the pickle is the restart fallback, absent is fine
    cp, _ = _load_metrics_pkls(cp_dir, eps, cp_results)
    # near-shell scores only the augmented arm, and only where its json exists
    ns = {ep: None for ep in eps}
    if nearshell_dir:
        for ep in eps:
            path = os.path.join(nearshell_dir, f'nested_rf_{ep}.json')
            if os.path.exists(path):
                with open(path) as fh:
                    ns[ep] = json.load(fh)['nested']

    def _win(ep, arm):
        """Best (value, method) for one arm of one endpoint; RF always competes."""
        cand = {'RF': rf[ep][arm + '_metrics'][metric]}
        if arm + '_metrics' in cp.get(ep, {}):
            cand['cp'] = cp[ep][arm + '_metrics'][metric]
        if arm == 'augmented_cv' and ns[ep]:
            cand['ns'] = ns[ep][ns_key]
        best = max(cand, key=cand.get)
        return cand[best], best

    # one row per endpoint: the winning value and the winning method for each of the three arms
    rows = []
    for ep in eps:
        row = {'n': rf[ep]['internal_cv_metrics']['n_test']}
        for arm in _BAR_ARMS:
            row[arm], row[arm + '_by'] = _win(ep, arm)
        row['note'] = (cp[ep]['_grouping'] if row['augmented_cv_by'] == 'cp'
                       else ('dist060_agree' if row['augmented_cv_by'] == 'ns' else 'all public'))
        rows.append(row)
    df = pd.DataFrame(rows, index=eps)

    # clip only what is DRAWN; the label keeps the true number (marked when floored)
    clip = (metric == 'r2det') and clip0
    C = params.SERAC_C
    colors = [C['paper'], C['ember'], C['azure']]
    x = np.arange(len(eps)); w = 0.27
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    for i, (arm, label) in enumerate(_BAR_ARMS.items()):
        true = df[arm].to_numpy(float)
        drawn = np.clip(true, 0.0, None) if clip else true
        bars = ax.bar(x + (i - 1) * w, drawn, w, label=label, color=colors[i], edgecolor='#333', lw=.6)
        ax.bar_label(bars, labels=[(f'↓{t:.2f}' if (clip and t < 0) else f'{t:.2f}') for t in true],
                     fontsize=7.5, padding=1.5)
    # limits from the drawn values; squared Pearson keeps the fixed 0-1.12 frame
    vals = [v2 for arm in _BAR_ARMS for v2 in (np.clip(df[arm], 0, None) if clip else df[arm]) if np.isfinite(v2)]
    if metric == 'r2':
        lo, hi, ny = 0.0, 1.12, 1.045
    else:
        pad = 0.14 * (max(vals) - min(vals))
        lo, hi, ny = min(0.0, min(vals)) - 0.4 * pad, max(vals) + 1.6 * pad, max(vals) + 0.35 * pad
    # zero reference line whenever a bar goes below it
    if lo < 0:
        ax.axhline(0, color='#444', lw=0.8, zorder=1)
    # per-endpoint internal test size across the top
    for xi, n in zip(x, df['n']):
        ax.annotate(f'n={int(n)}', (xi, ny), ha='center', fontsize=8, color='grey')
    ax.set_xticks(x); ax.set_xticklabels(eps, rotation=12)
    ax.set_ylabel(_METRIC_LABEL[metric] + (' — floored at 0' if clip else '')); ax.set_ylim(lo, hi)
    ax.grid(axis='y', alpha=.25); ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    # legend just above the axes; the title pad then clears it
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.005), frameon=False)
    ax.set_title(f'ADME endpoint prediction — {_METRIC_SHORT[metric]} by arm (internal target)',
                 pad=34, fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_png, dpi, f'bar_summary_RF_CP_{metric}.png', v)
    if v:
        show = df.copy()
        for arm in _BAR_ARMS:
            show[arm + '_by'] = show[arm + '_by'].map(LBL)
        print(f'winner per arm, {metric} (no internal compound in any transfer training set)')
        print(show.round(3).to_string())
    return fig, df


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
    param bool v: print how many compounds matched, and which endpoints were skipped
    return tuple: (long [compound, endpoint, real_y, pred_y, conf, resid], metrics indexed by endpoint).
        An endpoint with no prediction column is SKIPPED, so the metrics index holds only what the
        table actually predicts — check `.index` before indexing it by an endpoint name.
    """
    import ML_Reg                                    # lazy: only the notebook/runner path needs it
    rt = truth.merge(preds, left_on=truth_id, right_on=id_col, how='inner')
    if v:
        print(f'> round trip: {len(rt)} of {len(truth)} internal compounds matched a prediction')
    cols = {'R2_det': 'r2det', 'R2_pears': 'r2', 'RMSE': 'rmse', 'N': 'n_test'}
    long, metrics, skipped = [], [], []
    for k, ep in endpoints.items():
        # an external table need not predict every config endpoint -> score what it has, name the rest
        if pred_col(k) not in rt.columns:
            skipped.append(k)
            continue
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
    if skipped and v:
        print(f'> no prediction column for {skipped} — those endpoints are absent from the result')
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
