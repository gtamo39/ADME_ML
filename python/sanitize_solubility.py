"""Sanitize the public solubility data (remove physically-impossible values) and re-run a single-task
RF on solubility, non-destructively, for before/after comparison.

Root cause (2026-07-20 audit): the harmonized public solubility set (`public_solubility.parquet`,
value = log10 µM) carries a contaminated high tail — apparent solubilities up to 4.3e8 µM = 426 mol/L
(water is ~55 mol/L). Spread across AQUA/PHYS/PharmaBench/ChEMBL/ESOL (bad source logS/units), NOT one
source. In log space it sits at log10 6–8.6, 4+ log-units past the internal max (4.25); it poisons
training and produced a 317,000 µM raw-trained RMSE.

Fix: drop rows with solubility > SOL_MAX (1 mol/L = 1e6 µM = log10 6.0) — physically impossible for
these compound classes; ~0.7% of rows. Everything writes to *_solclean/ dirs; the production cache,
outputs, and deployed models are never touched.

Run (env ML, overnight):  python python/sanitize_solubility.py
  -> autoresearch/predict_adme_solclean/   (sanitized cache)
  -> output/predictions_runs_solclean/     (RF pred_dfs + reports + comparison)
Aggregate output only — no SMILES / per-compound values.
"""
from __future__ import annotations
import shutil, sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
CLEAN_CACHE = ROOT / 'autoresearch/predict_adme_solclean'
LOG_OUT = ROOT / 'output/predictions_runs'                 # existing contaminated solubility results (for comparison)
CLEAN_OUT = ROOT / 'output/predictions_runs_solclean'
CONFIG = ROOT / 'config/config.yaml'
SOL_MAX_LOG10_UM = 6.0                                     # 1 mol/L; solubility above this is physically impossible
sys.path.insert(0, os.path.expanduser('~/Scripts'))
from Statistics_tools import rsquared as rsq


def contamination_report():
    """Per-origin solubility value stats (log10 µM) + impossible-count; save + return."""
    d = pd.read_parquet(CACHE / 'public_solubility.parquet', columns=['value', 'origin'])
    g = d.groupby('origin')['value']
    imp = d[d['value'] > SOL_MAX_LOG10_UM].groupby('origin').size()
    tab = pd.DataFrame({'n': g.size(), 'median_log': g.median().round(2), 'p99_log': g.quantile(.99).round(2),
                        'max_log': g.max().round(2), 'raw_max_uM': (10 ** g.max()).map(lambda x: float(f'{x:.3g}')),
                        'n_impossible': imp.reindex(g.size().index).fillna(0).astype(int)}).sort_values('max_log', ascending=False)
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(CLEAN_OUT / 'solubility_contamination_report.csv')
    return tab


def build_clean_cache():
    """Sanitized cache: filter public_solubility to solubility <= 1 mol/L; copy internal targets + features."""
    CLEAN_CACHE.mkdir(parents=True, exist_ok=True)
    shutil.copy(CACHE / 'internal_targets.parquet', CLEAN_CACHE / 'internal_targets.parquet')
    shutil.copy(CACHE / 'internal_MF.parquet', CLEAN_CACHE / 'internal_MF.parquet')
    sol = pd.read_parquet(CACHE / 'public_solubility.parquet')
    kept = sol[sol['value'] <= SOL_MAX_LOG10_UM].reset_index(drop=True)
    kept.to_parquet(CLEAN_CACHE / 'public_solubility.parquet', index=False)
    print(f'> sanitized public_solubility: {len(sol)} -> {len(kept)} rows '
          f'(dropped {len(sol) - len(kept)} = {100 * (len(sol) - len(kept)) / len(sol):.2f}% with solubility > 1 mol/L)', flush=True)
    return len(sol), len(kept)


def run_rf_solubility():
    """Run the single-task RF for solubility (4 arms) against the sanitized cache; no MLTrail."""
    sys.path.insert(0, str(ROOT / 'python'))
    import run_RF_SingleTask_systematic as rf
    rf.CACHE, rf.FEATDIR = CLEAN_CACHE, CLEAN_CACHE / '_feats'
    params = rf.PARAMS(CONFIG); params.endpoints = ['solubility']; params.output_dir = str(CLEAN_OUT.relative_to(ROOT))
    data = rf.DATA().load_all(params)
    rf.OUTPUT().run_endpoint(data, params, 'solubility', registry=None)


def _metrics(path, transform='log10'):
    """(R²_log, RMSE_log, R²_raw, RMSE_raw, n) for a pred_df; None if absent/too small."""
    if not Path(path).exists():
        return None
    d = pd.read_parquet(path).dropna(subset=['real_y', 'pred_y'])
    if len(d) < 3 or d['real_y'].nunique() < 2:
        return None
    r2l = float(rsq(d['real_y'], d['pred_y'])); rmsel = float(np.sqrt(((d['real_y'] - d['pred_y']) ** 2).mean()))
    yo, yp = 10.0 ** d['real_y'].to_numpy(), 10.0 ** d['pred_y'].to_numpy()      # -> µM
    r2r = float(rsq(yo, yp)); rmser = float(np.sqrt(((yo - yp) ** 2).mean()))
    return round(r2l, 3), round(rmsel, 4), round(r2r, 3), float(f'{rmser:.4g}'), len(d)


def compare():
    """Before (contaminated) vs after (sanitized) solubility RF, per arm, log + raw metrics."""
    arms = ['internal_temporal', 'augmented_temporal', 'internal_cv', 'augmented_cv']
    rows = []
    for arm in arms:
        for tag, base in [('contaminated', LOG_OUT), ('sanitized', CLEAN_OUT)]:
            m = _metrics(base / 'solubility' / f'pred_{arm}.parquet')
            rows.append({'arm': arm, 'data': tag,
                         'R2_log': m[0] if m else None, 'RMSE_log': m[1] if m else None,
                         'R2_raw': m[2] if m else None, 'RMSE_raw_uM': m[3] if m else None, 'n': m[4] if m else None})
    out = pd.DataFrame(rows)
    out.to_csv(CLEAN_OUT / 'solubility_before_after.csv', index=False)
    return out


if __name__ == '__main__':
    print('=== 1. contamination report (per origin) ===', flush=True)
    print(contamination_report().to_string(), flush=True)
    print('\n=== 2. build sanitized cache ===', flush=True)
    build_clean_cache()
    print('\n=== 3. run single-task RF on sanitized solubility (4 arms) ===', flush=True)
    run_rf_solubility()
    print('\n=== 4. BEFORE (contaminated) vs AFTER (sanitized) — solubility RF ===', flush=True)
    print('R2 = in-house squared Pearson; RMSE_log in log10(µM); RMSE_raw_uM in µM.\n', flush=True)
    print(compare().to_string(index=False), flush=True)
    print(f'\n> reports + pred_dfs in {CLEAN_OUT}  | sanitized cache in {CLEAN_CACHE}', flush=True)
