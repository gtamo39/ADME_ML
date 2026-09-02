#!/usr/bin/env python3
"""Autoresearch-style campaign: search a MENU of NVS instance-selection/weighting strategies for the subset
that MAXIMISES internal mdck R2det, using LightGBM as the fast workhorse, with periodic champion-RF checks.

Grounded in transfer-learning literature (instance selection + importance weighting to mitigate NEGATIVE
transfer; nearest-neighbour and classifier-based density-ratio weighting). Strategies covered:
  baselines            internal_only, all_nvs
  distance shell       dist_<tau>                         (keep NVS within a Tanimoto distance of the train fold)
  kNN local aug        knn_<k>                            (each internal cmpd brings its k closest public analogs)
  similarity weighting distw_<tau>, expw_<tau>            (importance weight NVS by similarity to internal)
  density-ratio weight clfw_<tau>                         (classifier internal-vs-NVS -> P(internal)/(1-P))
  support matching     range_iqr_<tau>, range_mm          (keep NVS whose label sits in the internal range)
  agreement filter     agree_<delta>_<tau>                (keep NVS an internal-trained model already predicts)
  uncertainty filter   lowunc_<tau>                       (drop high within-NVS tree-variance labels)
  outlier removal      noout_<tau>                        (drop physically implausible NVS label tails)
  combos               knn20_simw, dist060_agree

HONEST scoring: each strategy gets a grouped-CV R2det (descriptive), and a NESTED CV picks the best strategy on
inner folds and scores it on held-out outer folds (the number to trust). Every run also re-scores the top-K
strategies with the champion RF to confirm LightGBM rankings track. Crash-safe: results stream to disk per
strategy; `--resume` skips finished ones. Aggregate output only (no SMILES / per-compound values printed).

Run detached in the `ML` env:
  screen -S camp
  ~/miniconda3/envs/ML/bin/python python/nvs_campaign.py --config config/config.yaml --endpoint mdck \
      --outdir output/results/20260827_NVS_campaign
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser('~/Scripts'))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from python.ADME_build_ML import PARAMS, DATA, OUTPUT
from python.nvs_subset_search import NVSSubsetSearch, _pearson_r2, _iqr


def _r2det(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or np.ptp(y) == 0:
        return np.nan
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ---------- selection / weighting primitives (operate on a built NVSSubsetSearch `s`) ----------

def _mask_ids(s, mask):
    return [s.nvs_ids[j] for j in np.where(mask)[0]]


def sel_dist(s, tr, tau):
    """NVS within Tanimoto distance < tau of the train fold (leakage-free)."""
    return _mask_ids(s, s._dist_to_train(tr) < tau)


def sel_knn(s, tr, k):
    """Union of the k nearest NVS to each internal-train compound (local augmentation)."""
    rows = [s._int_pos[c] for c in tr]
    idx = set()
    for r in s._sim[rows]:
        idx.update(np.argpartition(-r, min(k, len(r) - 1))[:k].tolist())
    return [s.nvs_ids[j] for j in idx]


def sel_range(s, tr, tau, mode):
    """NVS within tau AND whose label sits inside the internal-train label range (support matching)."""
    yint = s._byid.loc[list(tr), 'label'].to_numpy(float)
    lo, hi = (np.percentile(yint, [25, 75]) if mode == 'iqr' else (yint.min(), yint.max()))
    d, ynvs = s._dist_to_train(tr), s.nvs['label'].to_numpy(float)
    return _mask_ids(s, (d < tau) & (ynvs >= lo) & (ynvs <= hi))


def sel_noout(s, tr, tau):
    """NVS within tau minus label outliers beyond internal median +- 3*IQR (drop implausible NVS tails)."""
    yint = s._byid.loc[list(tr), 'label'].to_numpy(float)
    med, iqr = float(np.median(yint)), _iqr(yint)
    d, ynvs = s._dist_to_train(tr), s.nvs['label'].to_numpy(float)
    return _mask_ids(s, (d < tau) & (ynvs >= med - 3 * iqr) & (ynvs <= med + 3 * iqr))


def sel_agree(s, tr, tau, delta):
    """NVS within tau whose label an internal-trained model already predicts to within delta (pseudo-label filter)."""
    add = sel_dist(s, tr, tau)
    if not add:
        return []
    m = s._make()
    m.fit(s._byid.loc[list(tr), s.feats].to_numpy(), s._byid.loc[list(tr), 'label'].to_numpy(float))
    pred = m.predict(s._byid.loc[add, s.feats].to_numpy())
    ya = s._byid.loc[add, 'label'].to_numpy(float)
    return [c for c, pr, y in zip(add, pred, ya) if abs(pr - y) < delta]


def sel_lowunc(s, tr, tau):
    """NVS within tau with below-median within-NVS tree-variance (keep the self-consistent labels)."""
    add = sel_dist(s, tr, tau)
    thr = np.nanmedian(s._unc)
    return [c for c in add if np.isfinite(s._unc[s._nvs_pos[c]]) and s._unc[s._nvs_pos[c]] < thr]


def sel_mnn(s, tr, k):
    """Mutual/symmetric kNN (Gowda & Krishna 1979): keep NVS j only when some internal i is BOTH among j's k
    nearest internal-train AND has j among ITS k nearest NVS. A reciprocal, denoised local augmentation."""
    rows = [s._int_pos[c] for c in tr]
    sub = s._sim[rows]                                       # (n_train_internal, n_nvs)
    kk = min(k, sub.shape[1] - 1)
    int_to_nvs = [set(np.argpartition(-r, kk)[:kk].tolist()) for r in sub]   # each internal's k nearest NVS
    keep = set()
    for i in range(len(rows)):
        for j in int_to_nvs[i]:
            col = sub[:, j]
            if i in np.argpartition(-col, min(k, len(col) - 1))[:k]:         # i among j's k nearest internal
                keep.add(j)
    return [s.nvs_ids[j] for j in keep]


def wilson_edit(s, tr, add, k=3, delta=1.0):
    """Wilson editing / ENN (Wilson 1972): drop each added NVS whose label disagrees by > delta with the mean
    label of its k nearest INTERNAL-train compounds. Leakage-free label denoising (uses train internal only)."""
    if not add:
        return []
    rows = [s._int_pos[c] for c in tr]
    yint = s._byid.loc[list(tr), 'label'].to_numpy(float)
    sub = s._sim[rows]
    keep = []
    for c in add:
        nn = np.argpartition(-sub[:, s._nvs_pos[c]], min(k, len(rows) - 1))[:k]
        if abs(float(s._byid.loc[c, 'label']) - yint[nn].mean()) < delta:
            keep.append(c)
    return keep


def w_sim(s, tr, add):
    """Importance weight = max Tanimoto similarity of each NVS to the train fold."""
    rows = [s._int_pos[c] for c in tr]
    return {c: float(s._sim[rows, s._nvs_pos[c]].max()) for c in add}


def w_expsim(s, tr, add, k=4.0):
    return {c: float(np.exp(k * v)) for c, v in w_sim(s, tr, add).items()}


def w_clf(s, tr, add):
    """Density-ratio weight P(internal)/(1-P) from a LightGBM classifier internal(1) vs added-NVS(0)."""
    if not add:
        return {}
    from lightgbm import LGBMClassifier
    Xi = s._byid.loc[list(tr), s.feats].to_numpy()
    Xn = s._byid.loc[add, s.feats].to_numpy()
    clf = LGBMClassifier(n_estimators=200, num_leaves=63, n_jobs=s.output.cfg['n_jobs'],
                         random_state=s.seed, verbose=-1)
    clf.fit(np.vstack([Xi, Xn]), np.r_[np.ones(len(Xi)), np.zeros(len(Xn))])
    p = np.clip(clf.predict_proba(Xn)[:, 1], 1e-3, 1 - 1e-3)
    return {c: float(pi / (1 - pi)) for c, pi in zip(add, p)}


def build_strategies():
    """Return the ordered strategy menu: name -> fn(s, tr) -> (add_ids, weight_dict|None, label_dict|None)."""
    S = []
    S.append(('internal_only', lambda s, tr: ([], None, None)))
    S.append(('all_nvs', lambda s, tr: (list(s.nvs_ids), None, None)))
    for tau in (0.50, 0.55, 0.60, 0.65, 0.70):
        S.append((f'dist_{tau:.2f}', lambda s, tr, _t=tau: (sel_dist(s, tr, _t), None, None)))
    for k in (1, 2, 3, 5, 20, 50):
        S.append((f'knn_{k}', lambda s, tr, _k=k: (sel_knn(s, tr, _k), None, None)))
    for k in (3, 5):
        S.append((f'mnn_{k}', lambda s, tr, _k=k: (sel_mnn(s, tr, _k), None, None)))
    S.append(('enn_0.60', lambda s, tr: (wilson_edit(s, tr, sel_dist(s, tr, 0.60)), None, None)))
    S.append(('knn1_enn', lambda s, tr: (wilson_edit(s, tr, sel_knn(s, tr, 1)), None, None)))
    S.append(('distw_0.70', lambda s, tr: ((a := sel_dist(s, tr, 0.70)), w_sim(s, tr, a), None)))
    S.append(('distw_1.00', lambda s, tr: ((a := list(s.nvs_ids)), w_sim(s, tr, a), None)))
    S.append(('expw_0.70', lambda s, tr: ((a := sel_dist(s, tr, 0.70)), w_expsim(s, tr, a), None)))
    S.append(('range_iqr_0.80', lambda s, tr: (sel_range(s, tr, 0.80, 'iqr'), None, None)))
    S.append(('range_mm_1.00', lambda s, tr: (sel_range(s, tr, 1.00, 'mm'), None, None)))
    S.append(('agree_1.0_0.80', lambda s, tr: (sel_agree(s, tr, 0.80, 1.0), None, None)))
    S.append(('agree_0.5_0.80', lambda s, tr: (sel_agree(s, tr, 0.80, 0.5), None, None)))
    S.append(('lowunc_0.70', lambda s, tr: (sel_lowunc(s, tr, 0.70), None, None)))
    S.append(('noout_0.80', lambda s, tr: (sel_noout(s, tr, 0.80), None, None)))
    S.append(('clfw_0.70', lambda s, tr: ((a := sel_dist(s, tr, 0.70)), w_clf(s, tr, a), None)))
    S.append(('knn20_simw', lambda s, tr: ((a := sel_knn(s, tr, 20)), w_sim(s, tr, a), None)))
    S.append(('dist060_agree', lambda s, tr: (sel_agree(s, tr, 0.60, 1.0), None, None)))
    return S


# focused tight-selection set for the RF-vs-LightGBM head-to-head compare
DEFAULT_COMPARE = ['internal_only', 'knn_1', 'knn_2', 'knn_3', 'dist_0.60', 'dist_0.65',
                   'mnn_3', 'mnn_5', 'enn_0.60', 'knn1_enn', 'dist060_agree']

# tight family that survived the compare -> the pool for the honest nested verdict (include internal_only as a choice)
NESTED_TIGHT = ['internal_only', 'enn_0.60', 'dist060_agree', 'dist_0.60', 'knn_1', 'dist_0.65']


# ---------- evaluation ----------

def grouped_eval(s, fn, folds):
    """Grouped-CV over internal: per fold, fit internal-train + strategy-selected NVS, predict held-out internal.
    Returns (r2_pearson, r2det, n_nvs_median, pred_df)."""
    parts, nadd = [], []
    for fi, (tr, te) in enumerate(folds, 1):
        add, w, lab = fn(s, tr)
        nadd.append(len(add))
        pr = s._fit_predict(tr, te, add, nvs_label=lab, nvs_weight=w)
        parts.append(pd.DataFrame({'compound': list(te), 'real_y': s._byid.loc[te, 'label'].to_numpy(float),
                                   'pred_y': pr, 'fold': fi}))
    pdf = pd.concat(parts, ignore_index=True)
    return _pearson_r2(pdf['real_y'], pdf['pred_y']), _r2det(pdf['real_y'], pdf['pred_y']), int(np.median(nadd)), pdf


def nested_select(s, strat_map, eligible, n_outer=5, n_inner=4):
    """Nested CV over the eligible strategies: pick the best strategy on each outer-train's inner CV (by R2det),
    score it on the held-out outer fold. Returns honest R2/R2det + the per-outer-fold winner."""
    outer = s._grouped_folds(s.int_ids, n_splits=n_outer)
    parts, winners = [], []
    for o, (otr, ote) in enumerate(tqdm(outer, desc='nested outer'), 1):
        inner = s._grouped_folds(otr, n_splits=n_inner, seed=s.seed + 1)
        scores = {}
        for name in tqdm(eligible, desc=f'inner {o}/{len(outer)}', leave=False):
            scores[name] = grouped_eval(s, strat_map[name], inner)[1]      # inner R2det
        best = max(scores, key=lambda x: (scores[x] if np.isfinite(scores[x]) else -np.inf))
        winners.append(best)
        add, w, lab = strat_map[best](s, otr)
        pr = s._fit_predict(otr, ote, add, nvs_label=lab, nvs_weight=w)
        parts.append(pd.DataFrame({'compound': list(ote), 'real_y': s._byid.loc[ote, 'label'].to_numpy(float),
                                   'pred_y': pr}))
        tqdm.write(f'  [nested {o}/{len(outer)}] picked {best} (inner R2det={scores[best]:.3f})')
    pdf = pd.concat(parts, ignore_index=True)
    return {'honest_r2': _pearson_r2(pdf['real_y'], pdf['pred_y']), 'honest_r2det': _r2det(pdf['real_y'], pdf['pred_y']),
            'fold_winners': winners}


def main():
    ap = argparse.ArgumentParser(description="NVS instance-selection campaign (LightGBM workhorse + RF checks).")
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--endpoint', default='mdck')
    ap.add_argument('--outdir', default='output/results/20260827_NVS_campaign')
    ap.add_argument('--learner', default='lgbm', help="workhorse: lgbm | rf50 | rf")
    ap.add_argument('--rf_check', default='rf', help="cross-check learner for the top-K (rf | rf50)")
    ap.add_argument('--top_k_rf', type=int, default=6, help="how many top strategies to re-score with the RF check")
    ap.add_argument('--nested_max_nvs', type=int, default=60000, help="exclude huge-subset strategies from nested")
    ap.add_argument('--min_n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--compare', action='store_true',
                    help="focused head-to-head: score --strategies under BOTH --compare_learners (R2 + R2det each)")
    ap.add_argument('--nested_only', action='store_true',
                    help="honest nested-CV verdict over --strategies (default the tight family) with --learner")
    ap.add_argument('--strategies', default='', help="comma list for --compare / --nested_only (else the default set)")
    ap.add_argument('--compare_learners', default='rf,lgbm', help="learners for --compare mode")
    args = ap.parse_args()

    # build the endpoint sets + the LightGBM-driven search
    params = PARAMS(args.config).load_params()
    data = DATA(); data.load_df_internal_exp_all(params)
    data.load_combine_dfs(params); data.build_MF_features(params)
    ep = args.endpoint
    getattr(data, f'build_ML_data_{ep}')()
    data.get_internal_public_sets(ep, min_n=args.min_n)
    data.select_best_combo_and_update(params, ep)
    output = OUTPUT(params)
    s = NVSSubsetSearch(data, output, params, endpoint=ep, seed=args.seed, learner=args.learner)
    print(f"> {ep}: internal={len(s.int_ids)} nvs={len(s.nvs_ids)} | learner={args.learner}", flush=True)

    strategies = build_strategies()
    strat_map = dict(strategies)
    folds = s._grouped_folds(s.int_ids)

    # names that will actually run this session -> only pay for uncertainty if a 'lowunc' strategy needs it
    if args.strategies:
        run_names = args.strategies.split(',')
    elif args.compare:
        run_names = DEFAULT_COMPARE
    elif args.nested_only:
        run_names = NESTED_TIGHT
    else:
        run_names = [n for n, _ in strategies]
    s._unc = s.nvs_uncertainty() if any('lowunc' in n for n in run_names) else None

    out = args.outdir
    os.makedirs(out, exist_ok=True)

    # ---- honest nested-CV verdict over a chosen strategy pool, under --learner (RF or lgbm) ----
    if args.nested_only:
        base = grouped_eval(s, strat_map['internal_only'], folds)[1]      # internal-only grouped-CV R2det
        nested = nested_select(s, strat_map, run_names)
        summary = {'endpoint': ep, 'learner': args.learner, 'pool': run_names,
                   'internal_only_r2det': round(base, 4), 'nested': nested}
        json.dump(summary, open(os.path.join(out, f'nested_{args.learner}_{ep}.json'), 'w'), indent=2, default=float)
        print(f"\n===== NESTED VERDICT ({ep}, {args.learner}) =====")
        print(f"pool                = {run_names}")
        print(f"internal_only R2det = {base:.3f}")
        print(f"HONEST nested R2det = {nested['honest_r2det']:.3f}  (r2={nested['honest_r2']:.3f})")
        print(f"fold winners        = {nested['fold_winners']}")
        print(f"  -> {'BEATS' if nested['honest_r2det'] > base else 'does NOT beat'} internal-only ({base:.3f})")
        print(f"\n-> {out}/nested_{args.learner}_{ep}.json")
        return

    # ---- focused head-to-head: each strategy under BOTH learners, R2 + R2det side by side ----
    if args.compare:
        learners = args.compare_learners.split(',')
        rows = []
        for name in tqdm(run_names, desc='compare'):
            rec = {'strategy': name}
            for L in learners:
                s.learner = L
                t = time.perf_counter()
                r2, r2det, n_nvs, _ = grouped_eval(s, strat_map[name], folds)
                rec['n_nvs'] = n_nvs
                rec[f'{L}_r2'], rec[f'{L}_r2det'] = round(r2, 4), round(r2det, 4)
                rec[f'{L}_sec'] = round(time.perf_counter() - t, 1)
            rows.append(rec)
            tqdm.write(f'  {name:16} n_nvs={rec["n_nvs"]:<6} '
                       + '  '.join(f'{L}: R2={rec[f"{L}_r2"]:.3f} R2det={rec[f"{L}_r2det"]:.3f}' for L in learners))
        tbl = pd.DataFrame(rows).sort_values(f'{learners[-1]}_r2det', ascending=False).reset_index(drop=True)
        tbl.to_csv(os.path.join(out, f'compare_{ep}.csv'), index=False)
        print(f"\n===== COMPARE (" + ep + ", augmented grouped-CV, " + '+'.join(learners) + ") =====")
        print(tbl.to_string(index=False))
        print(f"\n-> {out}/compare_{ep}.csv")
        return

    os.makedirs(os.path.join(out, 'preddfs'), exist_ok=True)
    csv = os.path.join(out, f'campaign_{ep}.csv')
    done = set(pd.read_csv(csv)['strategy']) if (args.resume and os.path.exists(csv)) else set()
    rows = pd.read_csv(csv).to_dict('records') if (args.resume and os.path.exists(csv)) else []

    # ---- descriptive grouped-CV per strategy (LightGBM), streamed to disk ----
    for name, fn in tqdm(strategies, desc=f'strategies ({args.learner})'):
        if name in done:
            continue
        t = time.perf_counter()
        try:
            r2, r2det, n_nvs, pdf = grouped_eval(s, fn, folds)
        except Exception as ex:
            print(f"  [skip] {name}: {ex}", flush=True); continue
        pdf.to_parquet(os.path.join(out, 'preddfs', f'{name}.parquet'), index=False)
        rows.append({'strategy': name, 'learner': args.learner, 'n_nvs_median': n_nvs,
                     'r2': round(r2, 4), 'r2det': round(r2det, 4), 'sec': round(time.perf_counter() - t, 1)})
        pd.DataFrame(rows).to_csv(csv, index=False)      # crash-safe: rewrite after each strategy
        tqdm.write(f'  {name:16} n_nvs={n_nvs:<7} R2={r2:.3f}  R2det={r2det:.3f}  ({rows[-1]["sec"]}s)')

    tbl = pd.DataFrame(rows).sort_values('r2det', ascending=False).reset_index(drop=True)
    internal_only = tbl.loc[tbl.strategy == 'internal_only', 'r2det']
    base = float(internal_only.iloc[0]) if len(internal_only) else np.nan

    # ---- champion-RF cross-check on the top-K (does the LightGBM ranking track?) ----
    topk = [n for n in tbl['strategy'].head(args.top_k_rf) if n != 'internal_only']
    s.learner = args.rf_check
    rf_rows = []
    for name in tqdm(topk, desc=f'RF check ({args.rf_check})'):
        _, r2det_rf, _, _ = grouped_eval(s, strat_map[name], folds)
        lg = float(tbl.loc[tbl.strategy == name, 'r2det'].iloc[0])
        rf_rows.append({'strategy': name, 'lgbm_r2det': lg, f'{args.rf_check}_r2det': round(r2det_rf, 4)})
        tqdm.write(f'  [RF check] {name:16} lgbm={lg:.3f}  {args.rf_check}={r2det_rf:.3f}')
    s.learner = args.learner
    pd.DataFrame(rf_rows).to_csv(os.path.join(out, f'rf_check_{ep}.csv'), index=False)

    # ---- honest nested-CV selection over the eligible (not-huge) strategies ----
    eligible = [n for n in strat_map if n != 'all_nvs'
                and float(tbl.loc[tbl.strategy == n, 'n_nvs_median'].iloc[0]) <= args.nested_max_nvs]
    nested = nested_select(s, strat_map, eligible)
    json.dump({'internal_only_r2det': base, 'nested': nested, 'eligible': eligible,
               'top_k_rf': rf_rows, 'args': vars(args)},
              open(os.path.join(out, f'summary_{ep}.json'), 'w'), indent=2, default=float)

    # ---- final report ----
    print("\n===== CAMPAIGN RESULTS (" + ep + ", grouped-CV " + args.learner + ") =====")
    print(tbl[['strategy', 'n_nvs_median', 'r2', 'r2det', 'sec']].to_string(index=False))
    print(f"\ninternal_only R2det = {base:.3f}")
    print(f"best strategy       = {tbl.strategy.iloc[0]}  (R2det {tbl.r2det.iloc[0]:.3f})")
    print(f"HONEST nested R2det = {nested['honest_r2det']:.3f}  (winners per fold: {nested['fold_winners']})")
    print(f"  -> {'BEATS' if nested['honest_r2det'] > base else 'does NOT beat'} internal-only ({base:.3f})")
    print(f"\nRF cross-check -> {out}/rf_check_{ep}.csv ; full summary -> {out}/summary_{ep}.json")


if __name__ == "__main__":
    main()
