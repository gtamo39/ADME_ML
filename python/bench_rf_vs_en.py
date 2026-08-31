#!/usr/bin/env python3
"""Benchmark champion RF vs ElasticNet fit time at the NVS feature width, to gauge higher-throughput subset search.

Fit time depends on matrix SHAPE, not the values, so this uses a SYNTHETIC same-width matrix (sparse binary,
mimicking fingerprints) — no real chemistry loaded. Times fit at growing row counts and extrapolates to --to rows.
Run: ~/miniconda3/envs/ML/bin/python python/bench_rf_vs_en.py --sizes 1000,4000,12000 --features 4469 --n_jobs 16 --to 273241
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor


def _xy(n, f, density=0.08, seed=0):
    """Synthetic sparse-binary features (fingerprint-like) + a learnable linear label. float32."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, f), dtype=np.float32) < density).astype(np.float32)
    w = rng.normal(size=f).astype(np.float32)
    y = (X @ w + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, y


def _time(model, X, y):
    t = time.perf_counter(); model.fit(X, y); return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='1000,4000,12000', help="comma row counts to time")
    ap.add_argument('--features', type=int, default=4469, help="feature width (H237=4469, H236=4269)")
    ap.add_argument('--n_jobs', type=int, default=16, help="RF threads (champion uses 32)")
    ap.add_argument('--to', type=int, default=273241, help="extrapolate fit time to this many rows")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]

    # champion RF (config RF_SINGLETASK.champion) vs ElasticNet vs SGD-elasticnet (fastest linear at scale)
    rf_kw = dict(n_estimators=200, max_depth=20, max_features=0.3, min_samples_leaf=2,
                 n_jobs=args.n_jobs, random_state=42)
    print(f"features={args.features}  rf n_jobs={args.n_jobs}\n", flush=True)
    rows = []
    for n in sizes:
        X, y = _xy(n, args.features)
        t_rf = _time(RandomForestRegressor(**rf_kw), X, y)
        t_en = _time(ElasticNet(alpha=1e-3, max_iter=1000, tol=1e-3), X, y)
        t_sgd = _time(SGDRegressor(penalty='elasticnet', alpha=1e-4, max_iter=20, tol=1e-3, random_state=42), X, y)
        rows.append((n, t_rf, t_en, t_sgd))
        print(f"n={n:>7}  RF={t_rf:8.2f}s  EN={t_en:7.2f}s  SGD={t_sgd:6.2f}s  "
              f"| RF/EN={t_rf/t_en:6.1f}x  RF/SGD={t_rf/t_sgd:6.1f}x", flush=True)

    # linear extrapolation from the largest measured size to --to rows
    n0, rf0, en0, sgd0 = rows[-1]
    k = args.to / n0
    print(f"\nextrapolated to {args.to} rows (x{k:.1f} from n={n0}):")
    print(f"  RF  ~ {rf0 * k / 60:6.1f} min\n  EN  ~ {en0 * k:6.1f} s\n  SGD ~ {sgd0 * k:6.1f} s")
    print(f"  => one full-NVS fit: EN is ~{rf0/en0:.0f}x, SGD ~{rf0/sgd0:.0f}x faster than the champion RF")


if __name__ == "__main__":
    main()
