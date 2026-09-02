#!/usr/bin/env python3
"""Generalize the NVS near-shell finding to ALL 8 ADME endpoints.

For each endpoint we ask the same question mdck answered: given the FULL public pool (every origin, not the
pre-selected best combo), does a tight, denoised near-shell of public compounds beat internal-only?
Per endpoint it runs:
  compare : DEFAULT_COMPARE strategies scored under BOTH RF and LightGBM (augmented grouped-CV, R2 + R2det).
  nested  : honest nested-CV verdict over NESTED_TIGHT with the champion RF (the deploy learner).

The 369k-row H237 feature matrix is built ONCE and reused across endpoints. Each endpoint writes its own
outputs and a cross-endpoint verdict table is printed at the end. Crash-safe: an endpoint whose nested JSON
already exists is skipped (`--resume`); aggregate output only (no SMILES / per-compound values printed).

Run detached in the `ML` env:
  screen -S nearshell
  ~/miniconda3/envs/ML/bin/python python/run_adme_nearshell.py --config config/config.yaml \
      --outdir output/results/20260831_ADME_nearshell --resume
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser('~/Scripts'))

import numpy as np
import pandas as pd

from python.ADME_build_ML import PARAMS, DATA, OUTPUT
from python.nvs_subset_search import NVSSubsetSearch
from python.nvs_campaign import (build_strategies, grouped_eval, nested_select,
                                 DEFAULT_COMPARE, NESTED_TIGHT)

ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']


def run_compare(s, strat_map, folds, names, learners):
    """Score each strategy under every learner (augmented grouped-CV). Return a rows list (one dict per strategy)."""
    rows = []
    for name in names:
        rec = {'strategy': name}
        for L in learners:
            s.learner = L
            t = time.perf_counter()
            r2, r2det, n_nvs, _ = grouped_eval(s, strat_map[name], folds)
            rec['n_pool'] = n_nvs
            rec[f'{L}_r2'], rec[f'{L}_r2det'] = round(r2, 4), round(r2det, 4)
            rec[f'{L}_sec'] = round(time.perf_counter() - t, 1)
        rows.append(rec)
    return rows


def run_endpoint(ep, data, output, params, out, args):
    """Build the full-pool search for one endpoint, run compare + nested, write outputs, return a summary dict."""
    # build the endpoint modelling frame + internal/public split
    getattr(data, f'build_ML_data_{ep}')()
    data.get_internal_public_sets(ep, min_n=args.min_n)
    s = NVSSubsetSearch(data, output, params, endpoint=ep, seed=args.seed, learner='rf', pool='all')
    s._unc = None                                                    # DEFAULT_COMPARE/NESTED_TIGHT never use lowunc
    n_int, n_pool = len(s.int_ids), len(s.nvs_ids)
    print(f"\n=== {ep}: internal={n_int} public_pool={n_pool} ===", flush=True)
    # no public data -> nothing to augment with
    if n_pool == 0:
        return {'endpoint': ep, 'n_internal': n_int, 'n_pool': 0, 'note': 'no public pool'}

    strat_map = dict(build_strategies())
    folds = s._grouped_folds(s.int_ids)

    # compare: RF + LightGBM side by side ('all_nvs' with pool='all' = the augmented-CV arm, all public)
    names = args.strategies.split(',') if args.strategies else DEFAULT_COMPARE
    tag = args.tag or 'compare'
    rows = run_compare(s, strat_map, folds, names, args.compare_learners.split(','))
    cmp_tbl = pd.DataFrame(rows).sort_values('rf_r2det', ascending=False).reset_index(drop=True)
    cmp_tbl.to_csv(os.path.join(out, f'{tag}_{ep}.csv'), index=False)
    base = float(cmp_tbl.loc[cmp_tbl.strategy == 'internal_only', 'rf_r2det'].iloc[0])
    best = cmp_tbl.iloc[0]
    rec = {'endpoint': ep, 'n_internal': n_int, 'n_pool': n_pool, 'internal_rf_r2det': round(base, 4),
           'best_compare_strategy': best.strategy, 'best_compare_rf_r2det': round(float(best.rf_r2det), 4)}
    # augmented-CV arm, when it was part of this run
    if 'all_nvs' in set(cmp_tbl.strategy):
        r = cmp_tbl.loc[cmp_tbl.strategy == 'all_nvs'].iloc[0]
        rec['augmented_rf_r2det'] = round(float(r.rf_r2det), 4)
        rec['augmented_lgbm_r2det'] = round(float(r.get('lgbm_r2det', np.nan)), 4)
    if args.skip_nested:
        return rec

    # nested: honest RF verdict over the tight family
    s.learner = 'rf'
    nested = nested_select(s, strat_map, NESTED_TIGHT)
    json.dump({'endpoint': ep, 'learner': 'rf', 'pool': NESTED_TIGHT,
               'internal_only_r2det': round(base, 4), 'nested': nested},
              open(os.path.join(out, f'nested_rf_{ep}.json'), 'w'), indent=2, default=float)

    return {**rec,
            'nested_rf_r2det': round(nested['honest_r2det'], 4),
            'beats_internal': bool(nested['honest_r2det'] > base),
            'fold_winners': nested['fold_winners']}


def main():
    ap = argparse.ArgumentParser(description="ADME near-shell generalization across all 8 endpoints (RF + LightGBM).")
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--outdir', default='output/results/20260831_ADME_nearshell')
    ap.add_argument('--endpoints', default='all', help="comma list or 'all'")
    ap.add_argument('--compare_learners', default='rf,lgbm')
    ap.add_argument('--strategies', default='',
                    help="comma list to score (default DEFAULT_COMPARE); 'all_nvs' = the augmented-CV arm (all public)")
    ap.add_argument('--tag', default='', help="output filename prefix (default 'compare')")
    ap.add_argument('--skip_nested', action='store_true', help="score the strategies only; no nested verdict")
    ap.add_argument('--summary_name', default='summary_all.csv', help="cross-endpoint summary filename")
    ap.add_argument('--min_n', type=int, default=200, help="min public rows per origin kept by get_internal_public_sets")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true', help="skip endpoints whose nested_rf_<ep>.json already exists")
    args = ap.parse_args()

    eps = ENDPOINTS if args.endpoints == 'all' else args.endpoints.split(',')
    out = args.outdir
    os.makedirs(out, exist_ok=True)

    # build the internal + public frames and the H237 feature matrix ONCE (reused across endpoints)
    params = PARAMS(args.config).load_params()
    data = DATA(); data.load_df_internal_exp_all(params)
    data.load_combine_dfs(params); data.build_MF_features(params)
    output = OUTPUT(params)

    summ_path = os.path.join(out, args.summary_name)
    summ = pd.read_csv(summ_path).to_dict('records') if (args.resume and os.path.exists(summ_path)) else []
    done = {r['endpoint'] for r in summ}

    for ep in eps:
        # an endpoint counts as done when its summary row exists (plus its nested JSON, unless nested is skipped)
        if args.resume and ep in done and (args.skip_nested or os.path.exists(os.path.join(out, f'nested_rf_{ep}.json'))):
            print(f"[skip] {ep}: already done", flush=True); continue
        try:
            rec = run_endpoint(ep, data, output, params, out, args)
        except Exception as ex:
            print(f"[error] {ep}: {ex}", flush=True); continue
        summ = [r for r in summ if r.get('endpoint') != ep] + [rec]
        pd.DataFrame(summ).to_csv(summ_path, index=False)          # crash-safe after each endpoint
        if 'nested_rf_r2det' in rec:
            print(f"  {ep}: internal={rec['internal_rf_r2det']:.3f}  best={rec['best_compare_strategy']}"
                  f"({rec['best_compare_rf_r2det']:.3f})  nested={rec['nested_rf_r2det']:.3f}  "
                  f"{'BEATS' if rec['beats_internal'] else 'no-beat'}", flush=True)

    # cross-endpoint verdict
    tbl = pd.DataFrame(summ)
    cols = [c for c in ['endpoint', 'n_internal', 'n_pool', 'augmented_rf_r2det', 'internal_rf_r2det',
                        'best_compare_strategy', 'best_compare_rf_r2det', 'nested_rf_r2det',
                        'beats_internal'] if c in tbl.columns]
    print("\n===== ADME NEAR-SHELL — CROSS-ENDPOINT VERDICT (RF, honest nested) =====")
    print(tbl[cols].to_string(index=False))
    print(f"\n-> {summ_path}  (+ {args.tag or 'compare'}_<ep>.csv / nested_rf_<ep>.json per endpoint)")


if __name__ == "__main__":
    main()
