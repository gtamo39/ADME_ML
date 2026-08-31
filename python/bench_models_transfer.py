#!/usr/bin/env python3
"""Train on the ENTIRE NVS pool, predict internal — compare model families on compute time + R2 / R2det.

Transfer setup (mdck pilot): X_train = all NVS features, y_train = NVS labels; X_test = internal features,
y_test = internal labels. For each model, records fit seconds, R2 (squared Pearson), R2det (coeff. of
determination). Runs in the user's kernel/env (needs the real feature matrix — not loaded by the assistant).

Run:
  ~/miniconda3/envs/ML/bin/python python/bench_models_transfer.py --config config/config.yaml --endpoint mdck
"""
import argparse
import os
import sys
import time

# make repo root importable (so `import python.*` resolves) regardless of launch cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser('~/Scripts'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from python.ADME_build_ML import PARAMS, DATA

COL2RM = ['compound', 'smiles', 'label', 'source', 'origin', '_ik']


def _r2_pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1] ** 2) if (np.ptp(y) > 0 and np.ptp(p) > 0) else np.nan


def _r2det(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--endpoint', default='mdck')
    ap.add_argument('--n_jobs', type=int, default=32)
    ap.add_argument('--min_n', type=int, default=1000)
    args = ap.parse_args()

    # build the endpoint sets
    params = PARAMS(args.config).load_params()
    data = DATA(); data.load_df_internal_exp_all(params)
    data.load_combine_dfs(params); data.build_MF_features(params)
    ep = args.endpoint
    getattr(data, f'build_ML_data_{ep}')()
    data.get_internal_public_sets(ep, min_n=args.min_n)
    data.select_best_combo_and_update(params, ep)

    feats = [c for c in data.d.columns if c not in COL2RM]
    internal = data.internal
    nvs = data.pub[data.pub.origin.isin(data.combo)]
    # float32 to halve memory on the 273k x ~4469 matrix
    Xtr = nvs[feats].to_numpy(np.float32); ytr = nvs['label'].to_numpy(np.float32)
    Xte = internal[feats].to_numpy(np.float32); yte = internal['label'].to_numpy(np.float32)
    print(f"> {ep}: train(NVS)={Xtr.shape}  test(internal)={Xte.shape}", flush=True)

    ch = params.RF_SINGLETASK['champion']
    models = {
        'RF champion (200)': RandomForestRegressor(**ch, n_jobs=args.n_jobs, random_state=42),
        'RF small (50)':     RandomForestRegressor(**{**ch, 'n_estimators': 50}, n_jobs=args.n_jobs, random_state=42),
        'ElasticNet':        make_pipeline(StandardScaler(), ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=2000, tol=1e-3)),
    }
    # gradient-boosted trees: real LightGBM if installed, else sklearn's HistGradientBoosting (no install)
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                           subsample=0.8, colsample_bytree=0.6, n_jobs=args.n_jobs, random_state=42)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("  (LightGBM not installed -> using sklearn HistGradientBoosting; `pip install lightgbm` for LGBM)", flush=True)
        models['HistGBM (sklearn)'] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                                    max_leaf_nodes=63, random_state=42)

    rows = []
    for name, mdl in models.items():
        t = time.perf_counter(); mdl.fit(Xtr, ytr); fit_s = time.perf_counter() - t
        p = mdl.predict(Xte)
        rows.append({'model': name, 'fit_time_s': round(fit_s, 1),
                     'R2_pears': round(_r2_pearson(yte, p), 3), 'R2det': round(_r2det(yte, p), 3)})
        print(f"  {name:20} fit={fit_s:7.1f}s  R2={rows[-1]['R2_pears']}  R2det={rows[-1]['R2det']}", flush=True)

    tbl = pd.DataFrame(rows)[['model', 'fit_time_s', 'R2_pears', 'R2det']]
    print("\n" + tbl.to_string(index=False))
    out = f'output/results/bench_models_transfer_{ep}.csv'
    tbl.to_csv(out, index=False); print(f"\n> saved {out}")


if __name__ == "__main__":
    main()
