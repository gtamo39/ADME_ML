"""ADME public-data cleaning sweep — the solubility cleaning sweep generalized to the other endpoints.

Find, per endpoint, the public-augmentation cleaning strategy that maximizes single-task RF predictive
power on our internal ADME. Cleaning is applied ONLY to the PUBLIC set; the internal target and its
evaluation split are FIXED and identical across every strategy, so no cleaning can touch the test set:
  - temporal arm : internal newest-30% (by SRB id) of THAT endpoint = test; oldest-70% + cleaned public = train.
  - 5-fold CV arm: KFold(shuffle, seed) over internal compounds; each fold's internal holdout = test;
    the other internal folds + ALL cleaned public = train (public NEVER in a test fold).

Public comes in three source classes per endpoint (some absent): EXP (experimental — TDC/Biogen),
NVS (Novartis-NIBR predicted), ADM (ADMETlab predicted). The dominant lever mirrors solubility's
thermo-vs-kinetic: EXPERIMENTAL-vs-PREDICTED source matching. Value cleaners (physical clip, winsorize,
IQR, InChIKey dedup, internal-twin leak removal) are secondary.

Strategies (each = a source filter + a chain of value cleaners; degenerate ones auto-deduped per endpoint):
  S0_all_raw        : all available sources, no cleaning (naive baseline)
  S1_all_phys       : all sources + physical clip (drop values outside the endpoint's plausibility range)
  S2_exp_only       : EXPERIMENTAL sources only + physical clip           [collapses to internal-only if no EXP]
  S3_drop_adm       : drop ADMETlab (keep EXP+NVS) + physical clip
  S4_best_combo     : (drop ADM) + physical clip + InChIKey dedup + internal-twin removal + winsorize
  S5_internal_only  : no public (the floor every augmentation must beat)

Metrics per (endpoint, strategy, arm): calibration-sensitive R²_det (1−SS_res/SS_tot) — the ranking
metric — plus squared-Pearson r² (rank-only, affine-invariant → do NOT select on it), RMSE, bias,
Spearman, in modelling space AND raw units (back-transformed per endpoint).

Run (env `ML`):
  python python/adme_cleaning_sweep.py --audit-only                 # fast: per-source distribution vs internal
  python python/adme_cleaning_sweep.py [--endpoints logd,ppb] [--fast]
    -> output/predictions_runs_admesweep/adme_cleaning_sweep.csv  (+ per-(endpoint,strategy) pred parquets)
Aggregate output only — no SMILES / InChIKey values / per-compound values are ever printed.
"""
from __future__ import annotations
import argparse, sys, os
from functools import reduce
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
IKCACHE = CACHE / '_ikcache'                                 # compound->InChIKey cache per public parquet (compute once)
OUT = ROOT / 'output/predictions_runs_admesweep'
CONFIG = ROOT / 'config/config.yaml'
PUBLIC_SRC = {'EXP': 'public_{ep}.parquet', 'NVS': 'public_novartis_{ep}.parquet', 'ADM': 'public_admetlab_{ep}.parquet'}
sys.path.insert(0, os.path.expanduser('~/Scripts'))
import ML_Reg as ML_Reg
from Statistics_tools import rsquared as rsq
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------------------------------------------- InChIKeys (cached per parquet)
def _inchikeys(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append((Chem.MolToInchiKey(m) or None) if m else None)   # '' (empty/wildcard mol) -> None
    return np.array(out, dtype=object)


def _inchikeys_cached(tag, frame):
    """InChIKeys for frame[['compound','smiles']], cached to _ikcache/<tag>.parquet (recompute only if absent)."""
    IKCACHE.mkdir(parents=True, exist_ok=True)
    fp = IKCACHE / f'{tag}.parquet'
    if fp.exists():
        cache = pd.read_parquet(fp)
        if set(frame['compound']).issubset(set(cache['compound'])):
            return frame[['compound']].merge(cache, on='compound', how='left')['_ik'].to_numpy()
    ik = pd.DataFrame({'compound': frame['compound'].to_numpy(), '_ik': _inchikeys(frame['smiles'])})
    ik.to_parquet(fp, index=False)
    return ik['_ik'].to_numpy()


# ----------------------------------------------------------------- load per endpoint
def _feats():
    mf = pd.read_parquet(CACHE / 'internal_MF.parquet').drop_duplicates('compound')
    return mf, [c for c in mf.columns if c != 'compound']


def avail_sources(ep):
    return [s for s in PUBLIC_SRC if (CACHE / PUBLIC_SRC[s].format(ep=ep)).exists()]


def load(ep, mf, feats, cap, seed):
    """Internal (label+feats) and pooled public (all sources, tagged source/origin/_ik; predicted capped)."""
    tgt = pd.read_parquet(CACHE / 'internal_targets.parquet')[['compound', 'smiles', ep]].dropna(subset=[ep])
    internal = tgt.rename(columns={ep: 'label'}).merge(mf, on='compound')
    internal['_ik'] = _inchikeys_cached(f'internal_{ep}', internal)
    pubs = []
    for s in avail_sources(ep):
        p = pd.read_parquet(CACHE / PUBLIC_SRC[s].format(ep=ep))[['compound', 'smiles', 'value', 'origin'] + feats].rename(columns={'value': 'label'})
        if s in ('NVS', 'ADM') and len(p) > cap:                     # cap predicted sources for sweep tractability
            p = p.sample(n=cap, random_state=seed).reset_index(drop=True)
        p['source'] = s
        p['_ik'] = _inchikeys_cached(f'{PUBLIC_SRC[s].format(ep=ep)}', p)
        pubs.append(p)
    pub = pd.concat(pubs, ignore_index=True) if pubs else pd.DataFrame(columns=['compound', 'smiles', 'label', 'origin', 'source', '_ik'] + feats)
    internal_iks = set(pd.unique(internal['_ik'])); internal_iks.discard(None)
    # fairness invariants — a strategy must never be able to alter the fixed internal test
    assert internal['compound'].is_unique, f'{ep}: internal compound ids not unique'
    assert set(internal['compound']).isdisjoint(set(pub['compound'])), f'{ep}: public/internal id collision'
    assert internal[feats].isna().to_numpy().sum() == 0, f'{ep}: internal features contain NaN'
    assert len(pub) == 0 or pub[feats].isna().to_numpy().sum() == 0, f'{ep}: public features contain NaN'
    return internal, pub, internal_iks


# ----------------------------------------------------------------- source filters + value cleaners
def keep_all(p):        return p
def keep_exp(p):        return p[p['source'] == 'EXP']
def keep_no_adm(p):     return p[p['source'] != 'ADM']
def keep_none(p):       return p.iloc[0:0]

SOURCE_FILTER = {'all': keep_all, 'exp': keep_exp, 'no_adm': keep_no_adm, 'none': keep_none}


def clip_physical(p, lo, hi, **_):      return p[(p['label'] >= lo) & (p['label'] <= hi)]
def winsorize(p, wlo, whi, **_):
    if not len(p): return p
    q = p['label'].quantile([wlo, whi]); p = p.copy(); p['label'] = p['label'].clip(q.iloc[0], q.iloc[1]); return p
def iqr_outlier(p, k, **_):
    def _k(g):
        q1, q3 = g['label'].quantile(.25), g['label'].quantile(.75); iqr = q3 - q1
        return g[(g['label'] >= q1 - k * iqr) & (g['label'] <= q3 + k * iqr)]
    return p.groupby('origin', group_keys=False).apply(_k) if len(p) else p
def dedup_inchikey(p, **_):
    if not len(p): return p
    p = p[p['_ik'].notna()].copy(); p['label'] = p.groupby('_ik')['label'].transform('median'); return p.drop_duplicates('_ik')
def remove_internal_leak(p, internal_iks=frozenset(), **_):
    return p[~p['_ik'].isin(internal_iks)]

# strategy = (source_filter_key, [value-cleaner names in order])
STRATEGIES = {
    'S0_all_raw':       ('all',    []),
    'S1_all_phys':      ('all',    ['clip']),
    'S2_exp_only':      ('exp',    ['clip']),
    'S3_drop_adm':      ('no_adm', ['clip']),
    'S4_best_combo':    ('no_adm', ['clip', 'dedup', 'noleak', 'winsor']),
    'S5_internal_only': ('none',   []),
}
CLEANERS = {'clip': clip_physical, 'winsor': winsorize, 'iqr': iqr_outlier, 'dedup': dedup_inchikey, 'noleak': remove_internal_leak}


# ----------------------------------------------------------------- evaluation (RF; internal test FIXED)
def temporal_test_ids(internal):
    srb = internal['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
    return set(internal['compound'].to_numpy()[np.argsort(srb)[int(len(internal) * 0.7):]])


def _rf(rf_kw, fast):
    kw = dict(rf_kw)
    if fast: kw['n_estimators'] = 80
    return RandomForestRegressor(**kw)


def _eval(ML, id_sets, rf_kw, fast):
    _, pred = ML_Reg.K_fold_by_defined_IDs(ML, ID='compound', ID_sets=id_sets, model=_rf(rf_kw, fast),
                                           col_to_rm=['compound', 'smiles', 'label'], v=False)
    return pred


def _r2det(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float('nan') if ss == 0 else 1.0 - ((y - p) ** 2).sum() / ss


def _inv(transform, p):
    p = np.asarray(p, float)
    if transform == 'log10':     return 10.0 ** p
    if transform == 'logit_pct': return 100.0 / (1.0 + 10.0 ** (-p))     # -> % unbound
    return p                                                             # identity


def _metrics(pred, transform):
    d = pred.dropna(subset=['real_y', 'pred_y'])
    if len(d) < 3 or d['real_y'].nunique() < 2:
        return {}
    real, prd = d['real_y'].to_numpy(), d['pred_y'].to_numpy()
    yo, yp = _inv(transform, real), _inv(transform, prd)
    return {'R2det': round(_r2det(real, prd), 3),                        # ranking metric (calibration-sensitive)
            'RMSE': round(float(np.sqrt(((real - prd) ** 2).mean())), 4),
            'bias': round(float((prd - real).mean()), 3),
            'pearson_r2': round(float(rsq(real, prd)), 3),               # rank-only (affine-invariant; do NOT select on it)
            'spearman': round(float(spearmanr(real, prd).statistic), 3),
            'R2det_raw': round(_r2det(yo, yp), 3), 'RMSE_raw': float(f'{np.sqrt(((yo - yp) ** 2).mean()):.4g}'),
            'n_test': len(d)}


def clean_public(pub, strat, knobs, internal_iks):
    src_key, cleaner_names = STRATEGIES[strat]
    p = SOURCE_FILTER[src_key](pub)
    for nm in cleaner_names:
        p = CLEANERS[nm](p, internal_iks=internal_iks, **knobs)
    return p


def strat_signature(pub, strat, knobs, internal_iks):
    """(kept-source set, cleaner-name tuple) — identical signatures give identical fits (dedup degenerate strategies)."""
    src_key, cleaner_names = STRATEGIES[strat]
    kept = tuple(sorted(SOURCE_FILTER[src_key](pub)['source'].unique())) if len(pub) else ()
    return ((), ()) if not kept else (kept, tuple(cleaner_names))


def _run_or_load(ep, strat, arm, ML, id_sets, rf_kw, fast, resume):
    """Fit+save the arm's pred_df, or reload it on resume (metrics recomputed from real_y/pred_y)."""
    fp = OUT / ep / strat / f'pred_augmented_{arm}.parquet'
    if resume and fp.exists():
        try:
            p = pd.read_parquet(fp)
            if {'real_y', 'pred_y'}.issubset(p.columns) and len(p):
                return p
        except Exception:
            pass
    pred = _eval(ML, id_sets, rf_kw, fast)
    fp.parent.mkdir(parents=True, exist_ok=True)
    pred[['compound', 'real_y', 'pred_y']].to_parquet(fp, index=False)
    return pred


def run_endpoint(ep, mf, feats, params, fast, resume):
    knobs = {'lo': params['phys'][ep][0], 'hi': params['phys'][ep][1],
             'wlo': params['winsor'][0], 'whi': params['winsor'][1], 'k': params['iqr_k']}
    internal, pub, internal_iks = load(ep, mf, feats, params['cap'], params['seed'])
    transform = params['transform'][ep]
    cols = ['compound', 'smiles', 'label'] + feats
    test = temporal_test_ids(internal); ids = internal['compound'].to_numpy()
    folds = list(KFold(n_splits=params['cv_folds'], shuffle=True, random_state=params['seed']).split(ids))
    all_internal = internal['compound'].tolist()
    print(f'\n=== {ep} | internal={len(internal)} public_avail={ {s: int((pub["source"]==s).sum()) for s in pub["source"].unique()} } '
          f'transform={transform} phys={params["phys"][ep]} ===', flush=True)
    rows, seen = [], {}
    for strat in STRATEGIES:
        sig = strat_signature(pub, strat, knobs, internal_iks)
        if sig in seen:
            print(f'  {strat:18} == {seen[sig]} (same source set + cleaners) — aliased', flush=True)
            rows += [{**r, 'strategy': strat, 'alias_of': seen[sig]} for r in rows if r['strategy'] == seen[sig]]
            continue
        seen[sig] = strat
        pubc = clean_public(pub, strat, knobs, internal_iks)[cols] if len(pub) else pub.iloc[0:0][cols]
        ML = pd.concat([internal[cols], pubc], ignore_index=True)
        pub_ids = pubc['compound'].tolist()
        base = {'endpoint': ep, 'strategy': strat, 'n_public': len(pubc), 'kept_sources': '+'.join(sig[0]) or 'none', 'alias_of': ''}
        # arm 1: augmented temporal (internal newest-30% = test; oldest-70% + public = train)
        train_t = ML.loc[~ML['compound'].isin(test), 'compound'].tolist()
        pt = _run_or_load(ep, strat, 'temporal', ML, [[train_t, list(test)]], params['rf_kw'], fast, resume)
        rows.append({**base, 'arm': 'augmented_temporal', **_metrics(pt, transform)})
        # arm 2: augmented CV (internal folds; public always in train)
        pc = _run_or_load(ep, strat, 'cv', ML, [[list(ids[tr]) + pub_ids, list(ids[te])] for tr, te in folds],
                          params['rf_kw'], fast, resume)
        rows.append({**base, 'arm': 'augmented_cv', **_metrics(pc, transform)})
        # arm 3: PUBLIC-ONLY -> internal — train on cleaned public only, predict ALL internal (pure transfer). Skip if no public.
        if pub_ids:
            po = _run_or_load(ep, strat, 'public_only', ML, [[pub_ids, all_internal]], params['rf_kw'], fast, resume)
            rows.append({**base, 'arm': 'public_only', **_metrics(po, transform)})
        for arm in ('augmented_temporal', 'augmented_cv', 'public_only'):
            r = [x for x in rows if x['strategy'] == strat and x['arm'] == arm]
            if r:
                print(f'  {strat:18} {arm:18} R2det={r[0].get("R2det")} RMSE={r[0].get("RMSE")} bias={r[0].get("bias")} '
                      f'pearson_r2={r[0].get("pearson_r2")} R2det_raw={r[0].get("R2det_raw")} n_pub={r[0]["n_public"]}', flush=True)
    return rows


# ----------------------------------------------------------------- audit (no fits)
def audit(endpoints, mf, feats, params):
    print('=== ADME public-source AUDIT (modelling space; aggregate only) ===', flush=True)
    rows = []
    for ep in endpoints:
        internal, pub, internal_iks = load(ep, mf, feats, params['cap'], params['seed'])
        il = internal['label']; lo, hi = params['phys'][ep]
        rows.append({'endpoint': ep, 'source': 'INTERNAL', 'origin': '-', 'n': len(internal),
                     'median': round(il.median(), 2), 'p1': round(il.quantile(.01), 2), 'p99': round(il.quantile(.99), 2),
                     'min': round(il.min(), 2), 'max': round(il.max(), 2), 'n_out_phys': int(((il < lo) | (il > hi)).sum()),
                     'ik_leak_vs_int': '-', 'pct_ik_in_int': '-'})
        for s in pub['source'].unique() if len(pub) else []:
            g = pub[pub['source'] == s]; gl = g['label']; gik = set(pd.unique(g['_ik'])); gik.discard(None)
            rows.append({'endpoint': ep, 'source': s, 'origin': '|'.join(map(str, g['origin'].unique()))[:40], 'n': len(g),
                         'median': round(gl.median(), 2), 'p1': round(gl.quantile(.01), 2), 'p99': round(gl.quantile(.99), 2),
                         'min': round(gl.min(), 2), 'max': round(gl.max(), 2), 'n_out_phys': int(((gl < lo) | (gl > hi)).sum()),
                         'ik_leak_vs_int': int(g['_ik'].isin(internal_iks).sum()),
                         'pct_ik_in_int': round(100 * len(gik & internal_iks) / max(1, len(gik)), 1)})
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); tab.to_csv(OUT / 'adme_source_audit.csv', index=False)
    print(tab.to_string(index=False), flush=True)
    print(f'\n> wrote {OUT}/adme_source_audit.csv', flush=True)
    return tab


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', default=None, help='comma-separated subset (default = config ADME_CLEAN_SWEEP.endpoints)')
    ap.add_argument('--audit-only', action='store_true', help='per-source distribution vs internal; no RF fits')
    ap.add_argument('--fast', action='store_true', help='80 trees (quick smoke; ranking only)')
    ap.add_argument('--cap', type=int, default=None, help='override predicted_cap (smoke test)')
    ap.add_argument('--resume', action='store_true', help='reuse any already-saved arm pred_dfs (skip their re-fit)')
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    rf = cfg['RF_SINGLETASK']; sw = cfg['ADME_CLEAN_SWEEP']
    params = {'rf_kw': dict(rf['champion'], n_jobs=rf['n_jobs'], random_state=rf['seed']), 'seed': rf['seed'],
              'cv_folds': rf['cv_folds'], 'cap': args.cap or sw['predicted_cap'], 'winsor': sw['winsor_pct'], 'iqr_k': sw['iqr_k'],
              'phys': sw['phys_range'], 'transform': {ep: c['transform'] for ep, c in cfg['ADME_ENDPOINTS'].items()}}
    endpoints = (args.endpoints.split(',') if args.endpoints else sw['endpoints'])
    mf, feats = _feats()
    print(f'> {len(feats)} H236 feats | predicted_cap={params["cap"]} | endpoints={endpoints} | fast={args.fast}', flush=True)

    if args.audit_only:
        audit(endpoints, mf, feats, params); return

    all_rows = []
    for ep in endpoints:
        all_rows += run_endpoint(ep, mf, feats, params, args.fast, args.resume)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows).drop_duplicates(['endpoint', 'strategy', 'arm'])
    df.to_csv(OUT / 'adme_cleaning_sweep.csv', index=False)
    print('\n=== ADME CLEANING SWEEP — augmented_cv (decision arm) + public_only transfer, ranked by CV R²_det ===', flush=True)
    print('   public_only = train on cleaned public ONLY, predict ALL internal (pure domain transfer).', flush=True)
    for ep in endpoints:
        sub = df[df['endpoint'] == ep]
        cv = (sub[sub['arm'] == 'augmented_cv'][['strategy', 'kept_sources', 'n_public', 'R2det', 'bias']]
              .rename(columns={'R2det': 'cv_R2det', 'bias': 'cv_bias'}))
        po = (sub[sub['arm'] == 'public_only'][['strategy', 'R2det', 'bias']]
              .rename(columns={'R2det': 'pubonly_R2det', 'bias': 'pubonly_bias'}))
        m = cv.merge(po, on='strategy', how='left').sort_values('cv_R2det', ascending=False)
        print(f'\n-- {ep} --', flush=True)
        print(m.to_string(index=False), flush=True)
    print(f'\n> wrote {OUT}/adme_cleaning_sweep.csv', flush=True)


if __name__ == '__main__':
    main()
