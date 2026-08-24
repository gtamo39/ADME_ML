"""Solubility data-cleaning sweep — find the cleaning strategy that maximizes single-task RF
predictive power on our internal (thermodynamic) solubility.

Cleaning is applied ONLY to the PUBLIC augmentation set. The internal target and its evaluation split
are FIXED and identical across every strategy, so comparisons are fair and no cleaning can touch the
test set:
  - temporal arm: internal newest-30% (by SRB id) = test; oldest 70% + cleaned public = train.
  - 5-fold CV arm: KFold(shuffle, seed) over internal compounds; each fold's internal holdout = test;
    the other internal folds + ALL cleaned public = train (public NEVER in a test fold).

Strategies (each = a composition of cleaners on the PUBLIC frame):
  S0_contaminated          : none (current deployed baseline)
  S1_physical              : drop solubility > 1 mol/L (log10 µM > 6) — physically impossible
  S2_thermo                : S1 + keep only thermodynamic sources (internal is thermodynamic; 82% of public is kinetic)
  S3_winsor                : S1 + winsorize label to [p1, p99]
  S4_iqr                   : S1 + per-source Tukey IQR outlier removal (k=3)
  S5_dedup                 : S1 + collapse duplicate InChIKeys to their median label
  S6_noleak                : S1 + drop public rows whose InChIKey matches ANY internal compound (twin leakage)
  S7_combo                 : S1 + thermo + dedup + noleak + winsor (kitchen sink)
  S8_thermo_dedup_noleak   : S1 + thermo + dedup + noleak (no winsor)

Metrics per (strategy, arm): the CALIBRATION-SENSITIVE R²_det (coefficient of determination,
1−SS_res/SS_tot) AND squared-Pearson r² (rank-only), RMSE, and bias — in log10(µM) and raw µM —
plus Spearman. Ranked by CV R²_det (det, not Pearson: Pearson r² is affine-invariant and blind to
the ~+1.8 log10 shift kinetic augmentation injects, so it must NOT be the selection metric).

Run (env ML):  python python/solubility_cleaning_sweep.py [--strategies S1,S2,...] [--fast]
  -> output/predictions_runs_solsweep/solubility_cleaning_sweep.csv  (+ per-strategy pred parquets)
Aggregate output only — no SMILES / InChIKey values / per-compound values are ever printed.
"""
from __future__ import annotations
import argparse, sys, os
from functools import reduce
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
OUT = ROOT / 'output/predictions_runs_solsweep'
CONFIG = ROOT / 'config/config.yaml'
CLIP_LOG10_UM = 6.0                                  # 1 mol/L; solubility above this is physically impossible
sys.path.insert(0, os.path.expanduser('~/Scripts'))
import ML_Reg as ML_Reg
from Statistics_tools import rsquared as rsq
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from rdkit import Chem
import yaml


# ----------------------------------------------------------------- load once
def load():
    """Internal solubility (log10 µM) + H236 features, public solubility (+ precomputed InChIKeys)."""
    mf = pd.read_parquet(CACHE / 'internal_MF.parquet').drop_duplicates('compound')
    feats = [c for c in mf.columns if c != 'compound']
    tgt = pd.read_parquet(CACHE / 'internal_targets.parquet')[['compound', 'smiles', 'solubility']].dropna(subset=['solubility'])
    internal = tgt.rename(columns={'solubility': 'label'}).merge(mf, on='compound')
    pub = pd.read_parquet(CACHE / 'public_solubility.parquet').rename(columns={'value': 'label'})
    pub['_ik'] = _inchikeys(pub['smiles'])
    internal_iks = set(pd.unique(_inchikeys(internal['smiles'])))
    internal_iks.discard(None)
    # fairness invariants (else a strategy could silently alter the fixed internal test) — fail loud, not silent
    assert internal['compound'].is_unique, 'internal compound ids not unique'
    assert set(internal['compound']).isdisjoint(set(pub['compound'])), 'public/internal compound-id collision'
    assert internal[feats].isna().to_numpy().sum() == 0, 'internal features contain NaN (union-mean imputation would leak)'
    assert pub[feats].isna().to_numpy().sum() == 0, 'public features contain NaN (union-mean imputation would leak)'
    print(f'> loaded internal={len(internal)} public={len(pub)} feats={len(feats)} '
          f'internal_inchikeys={len(internal_iks)} | invariants OK', flush=True)
    return internal, pub, feats, internal_iks


def _inchikeys(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        out.append((Chem.MolToInchiKey(m) or None) if m else None)   # '' (empty/wildcard mol) -> None
    return np.array(out, dtype=object)


# ----------------------------------------------------------------- cleaners (operate on the public frame)
def clip_physical(p, **_):
    return p[p['label'] <= CLIP_LOG10_UM]


def thermo_only(p, **_):
    return p[p['origin'].astype(str).str.startswith('thermodynamic')]


def winsorize(p, **_):
    lo, hi = p['label'].quantile(0.01), p['label'].quantile(0.99)
    p = p.copy(); p['label'] = p['label'].clip(lo, hi); return p


def iqr_outlier(p, k=3.0, **_):
    def keep(g):
        q1, q3 = g['label'].quantile(0.25), g['label'].quantile(0.75); iqr = q3 - q1
        return g[(g['label'] >= q1 - k * iqr) & (g['label'] <= q3 + k * iqr)]
    return p.groupby('origin', group_keys=False).apply(keep)


def dedup_inchikey(p, **_):
    p = p[p['_ik'].notna()].copy()
    p['label'] = p.groupby('_ik')['label'].transform('median')       # resolve label conflicts by median
    return p.drop_duplicates('_ik')


def remove_internal_leak(p, internal_iks=frozenset(), **_):
    return p[~p['_ik'].isin(internal_iks)]                           # drop public twins of internal (verified 0 on current data)


BAD_SOURCES = ['thermodynamic:PHYS', 'thermodynamic:AQUA']          # 15.7% / 6.8% physically-impossible; far above internal range


def drop_bad_sources(p, **_):
    return p[~p['origin'].astype(str).isin(BAD_SOURCES)]


# NOTE (from the pre-run diagnostics on THIS data): internal↔public twin-leakage = 0, so remove_internal_leak
# removes 0 rows (kept in combos only as a safety net); all InChIKey duplicates are cross-assay thermo+kinetic
# pairs, so dedup is a no-op once thermo_only is applied. S6_noleak (== S1_physical here) was dropped.
STRATEGIES = {
    'S0_contaminated': [],
    'S1_physical': [clip_physical],
    'S2_thermo': [clip_physical, thermo_only],
    'S3_winsor': [clip_physical, winsorize],
    'S4_iqr': [clip_physical, iqr_outlier],
    'S5_dedup': [clip_physical, dedup_inchikey],
    'S7_combo': [clip_physical, thermo_only, dedup_inchikey, remove_internal_leak, winsorize],
    'S8_thermo_dedup_noleak': [clip_physical, thermo_only, dedup_inchikey, remove_internal_leak],
    'S9_thermo_pruned': [clip_physical, thermo_only, drop_bad_sources],   # predicted best: thermo minus the 2 worst sources
}


# ----------------------------------------------------------------- evaluation (RF; internal test FIXED)
def temporal_test_ids(internal):
    srb = internal['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
    return set(internal['compound'].to_numpy()[np.argsort(srb)[int(len(internal) * 0.7):]])


def _rf(params, fast):
    kw = dict(params['champion'], n_jobs=params['n_jobs'], random_state=params['seed'])
    if fast:
        kw['n_estimators'] = 80
    return RandomForestRegressor(**kw)


def _eval(ML, id_sets, params, fast):
    _, pred = ML_Reg.K_fold_by_defined_IDs(ML, ID='compound', ID_sets=id_sets, model=_rf(params, fast),
                                           col_to_rm=['compound', 'smiles', 'label'], v=False)
    return pred


def _r2det(y, p):
    """Coefficient of determination (1−SS_res/SS_tot) — calibration-SENSITIVE, unlike squared Pearson."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float('nan') if ss_tot == 0 else 1.0 - ((y - p) ** 2).sum() / ss_tot


def _metrics(pred):
    d = pred.dropna(subset=['real_y', 'pred_y'])
    if len(d) < 3 or d['real_y'].nunique() < 2:
        return {}
    real, prd = d['real_y'].to_numpy(), d['pred_y'].to_numpy()
    yo, yp = 10.0 ** real, 10.0 ** prd
    return {'R2det_log': round(_r2det(real, prd), 3),                 # ranking metric (calibration-sensitive)
            'RMSE_log': round(float(np.sqrt(((real - prd) ** 2).mean())), 4),
            'bias_log': round(float((prd - real).mean()), 3),
            'pearson_r2_log': round(float(rsq(real, prd)), 3),        # rank-only (affine-invariant; do NOT select on this)
            'spearman': round(float(spearmanr(real, prd).statistic), 3),
            'R2det_raw': round(_r2det(yo, yp), 3), 'RMSE_raw_uM': float(f'{np.sqrt(((yo - yp) ** 2).mean()):.4g}'),
            'n_test': len(d)}


def run_strategy(name, internal, pub, feats, internal_iks, params, fast):
    cols = ['compound', 'smiles', 'label'] + feats
    clean = reduce(lambda df, fn: fn(df, internal_iks=internal_iks), STRATEGIES[name], pub)
    pubc = clean[cols]
    ML = pd.concat([internal[cols], pubc], ignore_index=True)
    test = temporal_test_ids(internal)
    ids = internal['compound'].to_numpy()
    pub_ids = pubc['compound'].tolist()
    rows = []
    # temporal
    train_t = ML.loc[~ML['compound'].isin(test), 'compound'].tolist()
    pred = _eval(ML, [[train_t, list(test)]], params, fast)
    rows.append({'strategy': name, 'arm': 'augmented_temporal', 'n_public': len(pubc), 'n_train': len(train_t), **_metrics(pred)})
    _save(name, 'temporal', pred)
    # cv (internal folds; public always train)
    kf = KFold(n_splits=params['cv_folds'], shuffle=True, random_state=params['seed'])
    id_sets = [[list(ids[tr]) + pub_ids, list(ids[te])] for tr, te in kf.split(ids)]
    predcv = _eval(ML, id_sets, params, fast)
    rows.append({'strategy': name, 'arm': 'augmented_cv', 'n_public': len(pubc), 'n_train': len(ids) - len(ids) // params['cv_folds'] + len(pub_ids), **_metrics(predcv)})
    _save(name, 'cv', predcv)
    for r in rows:
        print(f"  {name:24} {r['arm']:18} R2det_log={r.get('R2det_log')} RMSE_log={r.get('RMSE_log')} "
              f"bias_log={r.get('bias_log')} pearson_r2={r.get('pearson_r2_log')} n_pub={r['n_public']}", flush=True)
    return rows


def _save(name, arm, pred):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    pred[['compound', 'real_y', 'pred_y']].to_parquet(d / f'pred_augmented_{arm}.parquet', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategies', default=','.join(STRATEGIES), help='comma-separated subset')
    ap.add_argument('--fast', action='store_true', help='80 trees (quick smoke; ranking only)')
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())['RF_SINGLETASK']
    params = {k: cfg[k] for k in ('champion', 'n_jobs', 'seed', 'cv_folds')}
    internal, pub, feats, internal_iks = load()
    names = [s for s in args.strategies.split(',') if s in STRATEGIES]
    all_rows = []
    for name in names:
        all_rows += run_strategy(name, internal, pub, feats, internal_iks, params, args.fast)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / 'solubility_cleaning_sweep.csv', index=False)
    cols = ['strategy', 'n_public', 'R2det_log', 'RMSE_log', 'bias_log', 'pearson_r2_log', 'spearman', 'R2det_raw']
    print('\n=== SOLUBILITY CLEANING SWEEP — ranked by R²_det (coefficient of determination — calibration-sensitive) ===', flush=True)
    print('   pearson_r2_log is rank-only (affine-invariant) — do NOT select on it. '
          'CV n_test = pooled out-of-fold (120); CV n_train is per-fold.', flush=True)
    for arm in ('augmented_cv', 'augmented_temporal'):
        sub = df[df['arm'] == arm].sort_values('R2det_log', ascending=False)
        print(f'\n-- {arm} (best R²_det first) --', flush=True)
        print(sub[cols].to_string(index=False), flush=True)
    print(f'\n> wrote {OUT}/solubility_cleaning_sweep.csv', flush=True)


if __name__ == '__main__':
    main()
