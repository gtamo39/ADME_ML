#!/usr/bin/env python3
"""CLI runner for the NVS subset search (notebook Cell A + B) — mdck pilot — with crash-safe incremental saves.

Runs the four levers over the NVS augmentation pool and writes, per arm, the OOF pred_df + a metrics dict
consumable by the notebook's `endpoint_metrics_table_from_dict`, plus the training-set membership so every
arm's training rows are reconstructable. Everything streams to disk as each arm finishes, so a dropped
connection never loses completed work; `--resume` skips arms already on disk.

Run in the `ML` env, detached so an SSH drop cannot kill it:
  screen -S nvs   # (or: nohup ... &)
  ~/miniconda3/envs/ML/bin/python python/run_nvs_cellab.py --config config/config.yaml \
      --outdir output/results/20260827_NVS_cellab --levers baseline,s1,s5,s3,s2,nested
  # detach: Ctrl-a d   |   reattach: screen -r nvs

Outputs (under --outdir):
  metrics_dict.pkl            {<arm>_preddf, <arm>_metrics}  -> endpoint_metrics_table_from_dict(d, 'mdck')
  preddfs/<arm>.parquet       each OOF pred_df (compound, real_y, pred_y, fold, residuals)
  summary.csv                 arm, r2, r2det, rmse, n_test, n_train, n_nvs_median
  train_membership.pkl        arm -> [{fold, n_int_train, n_nvs_added, test_ids}, ...]
  folds.pkl                   [{fold, internal_train, internal_test}]  (shared internal CV folds)
  nvs_fold_distance.parquet   [fold, compound, distance]  -> S1/S5/S3 NVS train at tau = {distance < tau}
  nvs_scaffold_groups.parquet [compound, scaffold_group]  (+ s2_chosen in arms.json)  -> S2 membership
  nvs_weights.parquet         [compound, uq_std, weight]  (S3)
  arms.json / config.json / README.txt
Aggregate stdout only — no SMILES / per-compound values printed.
"""
import argparse
import json
import os
import pickle
import sys

# make repo root importable (so `import python.*` resolves) no matter the launch cwd, and add shared Scripts
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.expanduser('~/Scripts'))

import numpy as np
import pandas as pd

from python.ADME_build_ML import PARAMS, DATA, OUTPUT
from python.nvs_subset_search import NVSSubsetSearch
import ML_Reg


def _save_dict(path, d):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def main():
    ap = argparse.ArgumentParser(description="Run the NVS subset search (Cell A+B) with incremental saves.")
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--endpoint', default='mdck')
    ap.add_argument('--outdir', default='output/results/20260827_NVS_cellab')
    ap.add_argument('--levers', default='baseline,s1,s5,s3,s2,nested',
                    help="comma subset of: baseline,s1,s5,s3,s2,nested")
    ap.add_argument('--taus', default=None, help="comma distance thresholds; default = quantile grid")
    ap.add_argument('--n_taus', type=int, default=8, help="grid size when --taus not given")
    ap.add_argument('--s5_methods', default='shift,affine', help="S5 correction(s): comma of shift,affine,anchor")
    ap.add_argument('--sim0', type=float, default=0.3, help="S5 'anchor' near-neighbour similarity threshold")
    ap.add_argument('--min_anchors', type=int, default=20, help="S5 'anchor' min pairs before it falls back to shift")
    ap.add_argument('--max_groups', type=int, default=8, help="S2 scaffold group count")
    ap.add_argument('--n_outer', type=int, default=5)
    ap.add_argument('--n_inner', type=int, default=4)
    ap.add_argument('--min_n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true', help="skip arms whose preddf parquet already exists")
    args = ap.parse_args()
    levers = set(x.strip() for x in args.levers.split(','))

    # ---- build the endpoint sets + the search object ----
    params = PARAMS(args.config).load_params()
    data = DATA(); data.load_df_internal_exp_all(params)
    data.load_combine_dfs(params); data.build_MF_features(params)
    output = OUTPUT(params)
    ep = args.endpoint
    getattr(data, f'build_ML_data_{ep}')()
    data.get_internal_public_sets(ep, min_n=args.min_n)
    data.select_best_combo_and_update(params, ep)
    search = NVSSubsetSearch(data, output, params, endpoint=ep, seed=args.seed)
    taus = [float(x) for x in args.taus.split(',')] if args.taus else search.distance_taus(n=args.n_taus)
    print(f"> {ep}: internal={len(search.int_ids)} nvs={len(search.nvs_ids)} | taus={ [round(t,3) for t in taus] }", flush=True)

    # ---- output scaffolding ----
    out = args.outdir
    os.makedirs(os.path.join(out, 'preddfs'), exist_ok=True)
    mpath = os.path.join(out, 'metrics_dict.pkl')
    metrics_dict = pickle.load(open(mpath, 'rb')) if (args.resume and os.path.exists(mpath)) else {}
    membership = (pickle.load(open(os.path.join(out, 'train_membership.pkl'), 'rb'))
                  if (args.resume and os.path.exists(os.path.join(out, 'train_membership.pkl'))) else {})
    summary = []
    arms_meta = {}

    # ---- shared membership tables (written once) ----
    folds = search._grouped_folds(search.int_ids)
    _save_dict(os.path.join(out, 'folds.pkl'),
               [{'fold': i + 1, 'internal_train': tr, 'internal_test': te} for i, (tr, te) in enumerate(folds)])
    # per-fold NVS distance-to-train -> reconstructs any tau's NVS training set as {distance < tau}
    dist_rows = []
    for i, (tr, _te) in enumerate(folds, 1):
        d = search._dist_to_train(tr)
        dist_rows.append(pd.DataFrame({'fold': i, 'compound': search.nvs_ids, 'distance': d}))
    pd.concat(dist_rows, ignore_index=True).to_parquet(os.path.join(out, 'nvs_fold_distance.parquet'), index=False)

    # ---- per-arm recorder (streams to disk after every arm) ----
    def add_arm(name, preddf_fn, record_list, n_train=None, extra=None):
        pth = os.path.join(out, 'preddfs', f'{name}.parquet')
        if args.resume and os.path.exists(pth):
            pdf = pd.read_parquet(pth)
            print(f"  [resume] {name}: loaded cached pred_df", flush=True)
        else:
            pdf = preddf_fn()
            pdf.to_parquet(pth, index=False)
        # metrics (n_train from the per-fold record if available)
        if n_train is None and record_list:
            n_train = int(np.median([r['n_int_train'] + r['n_nvs_added'] for r in record_list]))
        metrics_dict[name + '_preddf'] = pdf
        metrics_dict[name + '_metrics'] = ML_Reg.get_reg_metrics_from_preddf(pdf, ntrain=n_train or len(search.int_ids))
        if record_list:
            membership[name] = record_list
        m = metrics_dict[name + '_metrics']
        n_nvs = int(np.median([r['n_nvs_added'] for r in record_list])) if record_list else (extra or {}).get('n_nvs', 0)
        summary.append({'arm': name, 'r2': m.get('r2'), 'r2det': m.get('r2det'), 'rmse': m.get('rmse'),
                        'n_test': m.get('n_test'), 'n_train': m.get('n_train'), 'n_nvs_median': n_nvs})
        arms_meta[name] = extra or {}
        # stream everything to disk NOW (crash-safe)
        _save_dict(mpath, metrics_dict)
        _save_dict(os.path.join(out, 'train_membership.pkl'), membership)
        pd.DataFrame(summary).to_csv(os.path.join(out, 'summary.csv'), index=False)
        print(f"  [saved] {name}: r2={m.get('r2')} r2det={m.get('r2det')} rmse={m.get('rmse')} "
              f"n_train~{m.get('n_train')} n_nvs~{n_nvs}", flush=True)

    # ---- baselines ----
    if 'baseline' in levers:
        rec = []; add_arm('internal_only', lambda: search.cv_preddf(lambda tr: [], record=rec), rec)
        rec = []; add_arm('all_nvs', lambda: search.cv_preddf(lambda tr: search.nvs_ids, record=rec), rec)

    # ---- S1 distance sweep ----
    if 's1' in levers:
        for t in taus:
            rec = []
            add_arm(f'S1_tau{t:.3f}',
                    lambda _t=t, _r=rec: search.cv_preddf(lambda tr: search._select_within(tr, _t), record=_r),
                    rec, extra={'lever': 'S1', 'tau': t})

    # ---- S5 bias-correction (per method x tau) ----
    if 's5' in levers:
        for method in [m.strip() for m in args.s5_methods.split(',')]:
            lab = search._bias_label(method=method, sim0=args.sim0, min_anchors=args.min_anchors)
            for t in taus:
                rec = []
                add_arm(f'S5{method}_tau{t:.3f}',
                        lambda _l=lab, _t=t, _r=rec: search.cv_preddf(lambda tr: search._select_within(tr, _t),
                                                                      nvs_label_fn=_l, record=_r),
                        rec, extra={'lever': 'S5', 'method': method, 'tau': t, 'sim0': args.sim0})

    # ---- S3 label-uncertainty weighting (uncertainty computed once, reused across taus) ----
    if 's3' in levers:
        std = search.nvs_uncertainty()
        scale = float(np.nanmedian(std))
        w = {c: float(np.exp(-std[j] / scale)) if np.isfinite(std[j]) else 1.0 for j, c in enumerate(search.nvs_ids)}
        pd.DataFrame({'compound': search.nvs_ids, 'uq_std': std, 'weight': [w[c] for c in search.nvs_ids]}
                     ).to_parquet(os.path.join(out, 'nvs_weights.parquet'), index=False)
        for t in taus:
            rec = []
            add_arm(f'S3_tau{t:.3f}',
                    lambda _t=t, _r=rec: search.cv_preddf(lambda tr: search._select_within(tr, _t),
                                                          nvs_weight=w, record=_r),
                    rec, extra={'lever': 'S3', 'tau': t, 'scale': scale})

    # ---- S2 scaffold greedy (forward) ----
    if 's2' in levers:
        groups = search.scaffold_groups(max_groups=args.max_groups)
        pd.DataFrame([{'compound': c, 'scaffold_group': g} for g, ids in groups.items() for c in ids]
                     ).to_parquet(os.path.join(out, 'nvs_scaffold_groups.parquet'), index=False)
        res = search.scaffold_greedy(direction='forward', max_groups=args.max_groups)
        res['path'].to_csv(os.path.join(out, 's2_greedy_path.csv'), index=False)
        chosen_ids = [c for g in res['chosen'] for c in groups[g]]
        rec = []
        add_arm('S2_scaffold', lambda: search.cv_preddf(lambda tr: chosen_ids, record=rec), rec,
                extra={'lever': 'S2', 'chosen_groups': res['chosen'], 'cv_r2': res['r2']})

    # ---- nested honest S1 ----
    if 'nested' in levers:
        nd = search.nested_distance(taus=taus, n_outer=args.n_outer, n_inner=args.n_inner)
        add_arm('S1_nested', lambda: nd['preddf'], None, n_train=len(search.int_ids),
                extra={'lever': 'S1_nested', 'chosen_taus': nd['chosen_taus'], 'honest_r2': nd['honest_r2']})

    # ---- manifests ----
    json.dump({k: {kk: (list(vv) if isinstance(vv, (list, tuple)) else vv) for kk, vv in v.items()}
               for k, v in arms_meta.items()}, open(os.path.join(out, 'arms.json'), 'w'), indent=2, default=float)
    json.dump(vars(args), open(os.path.join(out, 'config.json'), 'w'), indent=2)
    with open(os.path.join(out, 'README.txt'), 'w') as f:
        f.write("NVS subset search (mdck) — load `metrics_dict.pkl` and call the notebook's\n"
                "endpoint_metrics_table_from_dict(metrics_dict, 'mdck') to get the metrics table.\n"
                "Per-arm OOF pred_dfs are in preddfs/. Training-set membership: folds.pkl (internal CV folds),\n"
                "nvs_fold_distance.parquet (NVS train at tau = rows with distance < tau, per fold),\n"
                "nvs_scaffold_groups.parquet + arms.json['S2_scaffold'].chosen_groups (S2), nvs_weights.parquet (S3),\n"
                "train_membership.pkl (per-arm per-fold counts + test ids).\n")
    print(f"> DONE. wrote {len(summary)} arms -> {out}", flush=True)


if __name__ == "__main__":
    main()
