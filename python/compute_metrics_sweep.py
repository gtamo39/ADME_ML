"""Sweep every saved pred_df and compute a core metric panel per (source, endpoint, arm).

Pearson-r2 (in-house = squared Pearson r, scale/offset-invariant) is complemented by:
  - r2_det : coefficient of determination (1 - SSres/SStot) — penalizes bias + wrong slope, goes <0
  - rmse   : root mean squared error (native units, absolute accuracy incl. bias)
  - mae    : mean absolute error (robust typical error)
  - spearman : rank correlation
  - calib_gap = pearson_r2 - r2_det : how much Pearson-r2 overstates vs a calibrated fit (large = the
    "correlated but biased / flattened" case where r2 alone is misleading).

Aggregate output only (no per-compound values). Reads pred_*.parquet under both run dirs.
Run (env ML):  python python/compute_metrics_sweep.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DIRS = {'RF': ROOT / 'output/predictions_runs', 'chemprop': ROOT / 'output/predictions_runs_chemprop'}
ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']


def regression_metrics(pred):
    """Core regression metrics for a pred_df (needs real_y, pred_y). None if too small/constant."""
    d = pred.dropna(subset=['real_y', 'pred_y'])
    y, yhat = d['real_y'].to_numpy(float), d['pred_y'].to_numpy(float)
    if len(d) < 3 or np.unique(y).size < 2:
        return {'n': len(d), 'pearson_r2': None, 'r2_det': None, 'rmse': None, 'mae': None, 'spearman': None, 'calib_gap': None}
    pear2 = float(np.corrcoef(y, yhat)[0, 1] ** 2)          # == in-house Statistics_tools.rsquared
    r2d = float(r2_score(y, yhat))
    return {'n': len(d), 'pearson_r2': round(pear2, 3), 'r2_det': round(r2d, 3),
            'rmse': round(float(np.sqrt(mean_squared_error(y, yhat))), 3),
            'mae': round(float(mean_absolute_error(y, yhat)), 3),
            'spearman': round(float(spearmanr(y, yhat).statistic), 3),
            'calib_gap': round(pear2 - r2d, 3)}


def main():
    rows = []
    for source, d in DIRS.items():
        for p in sorted(d.glob('*/pred_*.parquet')):
            rows.append({'source': source, 'endpoint': p.parent.name, 'arm': p.stem[len('pred_'):],
                         **regression_metrics(pd.read_parquet(p, columns=['real_y', 'pred_y']))})
    long = pd.DataFrame(rows)
    long = long.sort_values(['endpoint', 'source', 'arm'],
                            key=lambda s: s.map({e: i for i, e in enumerate(ENDPOINTS)}) if s.name == 'endpoint' else s).reset_index(drop=True)
    dest = ROOT / 'output/predictions_runs/metrics_all.csv'
    long.to_csv(dest, index=False)
    print(f'> wrote {dest}  ({len(long)} pred_dfs)\n')
    print(long.to_string(index=False))
    # spotlight: where Pearson-r2 most overstates a calibrated fit
    flag = long.dropna(subset=['calib_gap']).nlargest(10, 'calib_gap')
    print('\n=== largest calibration gaps (Pearson-r2 >> R2_det: correlated but biased/flattened) ===')
    print(flag[['source', 'endpoint', 'arm', 'pearson_r2', 'r2_det', 'rmse', 'calib_gap', 'n']].to_string(index=False))


if __name__ == '__main__':
    main()
