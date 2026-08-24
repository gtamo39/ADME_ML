"""Raw-units ablation: what if we modelled the ADME endpoints in their native CDD assay units
instead of log10/logit space (logD stays identity — already log)?

The log/logit transform lives entirely in the cached target parquets, NOT in the RF/chemprop
runners — so "redo in raw units" = rebuild the cache with back-transformed targets, then re-run the
SAME scripts against it. Everything writes to *_raw/ dirs; the production cache, outputs, and the
deployed chemprop checkpoints are never touched.

Steps (RF runs in env `ML`, chemprop in env `chemprop`):
  build_raw_cache() — back-transform internal_targets + every public_* parquet to raw units
                      (exact inverse of extract_adme._transform); features copied as-is.
  run_rf(eps)       — single-task RF, TEMPORAL arm, all 8 endpoints (caco2 = noADM, matching the log 'best').
  run_chemprop(grps)— multitask chemprop, TEMPORAL, best groupings (all8 + clearance).
  compare_winners() — per endpoint: R² log (reported) | log-trained scored in raw | raw-trained in raw.

Run:  python python/raw_units_experiment.py --rf              # env ML  (builds cache + 8 RF)
      python python/raw_units_experiment.py --chemprop        # env chemprop (all8 + clearance)
      python python/raw_units_experiment.py --compare         # env ML  (print comparison)
Aggregate output only — no SMILES / per-compound values.
"""
from __future__ import annotations
import argparse, shutil, sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
RAW_CACHE = ROOT / 'autoresearch/predict_adme_raw'
CONFIG = ROOT / 'config/config.yaml'
RF_OUT, CP_OUT = 'output/predictions_runs_raw', 'output/predictions_runs_chemprop_raw'
ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']
sys.path.insert(0, os.path.expanduser('~/Scripts'))
from Statistics_tools import rsquared as rsq

INV = {'identity': lambda y: y, 'log10': lambda y: np.power(10.0, y),
       'logit_pct': lambda y: 100.0 / (1.0 + np.power(10.0, -y))}      # -> % unbound


def _transforms():
    cfg = yaml.safe_load(CONFIG.read_text())['ADME_ENDPOINTS']
    return {ep: cfg[ep]['transform'] for ep in ENDPOINTS}


def _winners():
    return yaml.safe_load(CONFIG.read_text())['PUBLIC_TO_INTERNAL']['winners']


def _ep_of(fname):
    """Endpoint an EXP/NVS/ADM public parquet belongs to (None if not an endpoint file)."""
    stem = (fname.replace('public_novartis_', '').replace('public_admetlab_', '')
                 .replace('public_', '').replace('.parquet', ''))
    return stem if stem in ENDPOINTS else None


def build_raw_cache():
    """Materialize a raw-units twin of the cache: targets/values inverted to native assay units."""
    tr = _transforms()
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    tgt = pd.read_parquet(CACHE / 'internal_targets.parquet')
    for ep in ENDPOINTS:
        if ep in tgt:
            tgt[ep] = INV[tr[ep]](tgt[ep].to_numpy(float))             # back-transform each endpoint column
    tgt.to_parquet(RAW_CACHE / 'internal_targets.parquet', index=False)
    shutil.copy(CACHE / 'internal_MF.parquet', RAW_CACHE / 'internal_MF.parquet')   # features unchanged
    n = 0
    for f in CACHE.glob('public_*.parquet'):                          # back-transform each public 'value'
        ep = _ep_of(f.name)
        if ep is None:
            continue
        d = pd.read_parquet(f)
        if 'value' in d:
            d['value'] = INV[tr[ep]](d['value'].to_numpy(float))
        d.to_parquet(RAW_CACHE / f.name, index=False); n += 1
    print(f'> built raw cache at {RAW_CACHE} (internal_targets + internal_MF + {n} public parquets)', flush=True)


def run_rf(endpoints):
    """Single-task RF, TEMPORAL only, against the raw cache; pred_dfs -> RF_OUT/<ep>/."""
    sys.path.insert(0, str(ROOT / 'python'))
    import run_RF_SingleTask_systematic as rf
    rf.CACHE, rf.FEATDIR = RAW_CACHE, RAW_CACHE / '_feats'
    params = rf.PARAMS(CONFIG); params.endpoints = endpoints; params.output_dir = RF_OUT
    data = rf.DATA().load_all(params); out = rf.OUTPUT()
    for ep in endpoints:
        aug = data.augmented_sources(ep, params)
        if ep == 'caco2':
            aug = [s for s in aug if s != 'ADM']                      # match the log 'best' (ADMETlab hurts caco2)
        epdir = ROOT / RF_OUT / ep; epdir.mkdir(parents=True, exist_ok=True)
        for arm, srcs in [('internal_temporal', []), ('augmented_temporal', aug)]:
            pred = out.eval_temporal(data, params, ep, srcs)
            if pred is not None:
                pred.to_parquet(epdir / f'pred_{arm}.parquet', index=False)
        print(f'  {ep:11} temporal done (augmented sources={aug})', flush=True)


def run_chemprop(groupings):
    """Multitask chemprop, TEMPORAL, best groupings, against the raw cache; outputs -> CP_OUT.
    Redirects the model-save + wide-public paths so production checkpoints/cache are untouched."""
    sys.path.insert(0, str(ROOT / 'python'))
    import run_Chemprop_SystematicGroups as cps
    cps.CACHE = RAW_CACHE                                             # internal + EXP public reads
    cps.cp.HERE = RAW_CACHE                                           # _wide_public reads raw NVS/ADM
    cps.MODELS_DIR = ROOT / 'output/chemprop_models_raw'              # DO NOT clobber deployed checkpoints
    params = cps.PARAMS(CONFIG); params.output_dir = CP_OUT
    data = cps.DATA().load_all(params); out = cps.OUTPUT()
    rows, records = out.evaluate_all(data, params, groupings)
    out.write_outputs(data, params, rows, records)


def _rsq_rmse(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return round(float(rsq(real, pred)), 3), round(float(np.sqrt(((real - pred) ** 2).mean())), 4)


def _winner_paths(ep):
    """(log-space pred parquet, raw-trained pred parquet) for the endpoint's best model."""
    w = _winners()[ep]
    if w['model'] == 'chemprop':
        g = w['grouping']
        return (ROOT / f'output/predictions_runs_chemprop/{ep}/pred_{g}_temporal.parquet',
                ROOT / CP_OUT / ep / f'pred_{g}_temporal.parquet')
    arm = 'augmented_temporal_noADM' if ep == 'caco2' else 'augmented_temporal'
    logp = ROOT / f'output/predictions_runs/{ep}/{arm}.parquet'
    logp = logp if logp.exists() else ROOT / f'output/predictions_runs/{ep}/pred_{arm}.parquet'
    return logp, ROOT / RF_OUT / ep / 'pred_augmented_temporal.parquet'


def compare_winners():
    """Per endpoint (best model, temporal): R² in log space (reported) vs the same model back-transformed
    to raw vs a model RE-FIT on raw targets. Answers whether training in raw units helps the raw metric."""
    tr, win = _transforms(), _winners()
    rows = []
    for ep in ENDPOINTS:
        model = f"chemprop:{win[ep].get('grouping')}" if win[ep]['model'] == 'chemprop' else 'RF'
        logp, rawp = _winner_paths(ep)
        r2_log = r2_lt_raw = rmse_lt_raw = r2_rt_raw = rmse_rt_raw = n = None
        if logp.exists():
            d = pd.read_parquet(logp).dropna(subset=['real_y', 'pred_y'])
            if len(d) >= 3 and d['real_y'].nunique() > 1:
                r2_log = _rsq_rmse(d['real_y'], d['pred_y'])[0]
                r2_lt_raw, rmse_lt_raw = _rsq_rmse(INV[tr[ep]](d['real_y']), INV[tr[ep]](d['pred_y']))
        if rawp.exists():
            d = pd.read_parquet(rawp).dropna(subset=['real_y', 'pred_y'])
            if len(d) >= 3 and d['real_y'].nunique() > 1:
                r2_rt_raw, rmse_rt_raw = _rsq_rmse(d['real_y'], d['pred_y']); n = len(d)
        rows.append({'endpoint': ep, 'model': model, 'transform': tr[ep], 'n': n,
                     'R2_log': r2_log, 'R2_logtrained_raw': r2_lt_raw, 'R2_rawtrained_raw': r2_rt_raw,
                     'RMSE_logtrained_raw': rmse_lt_raw, 'RMSE_rawtrained_raw': rmse_rt_raw})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rf', action='store_true', help='env ML: build raw cache + run 8 single-task RF (temporal)')
    ap.add_argument('--chemprop', action='store_true', help='env chemprop: run all8 + clearance groupings (temporal)')
    ap.add_argument('--groupings', nargs='*', default=['all8', 'clearance'])
    ap.add_argument('--compare', action='store_true', help='env ML: print the log-vs-raw comparison table')
    args = ap.parse_args()
    if args.rf:
        build_raw_cache(); run_rf(ENDPOINTS)
    if args.chemprop:
        if not (RAW_CACHE / 'internal_targets.parquet').exists():
            build_raw_cache()
        run_chemprop(args.groupings)
    if args.compare or not (args.rf or args.chemprop):
        print('\n=== RAW-UNITS ablation — best model per endpoint (temporal) ===', flush=True)
        print('R2 = in-house squared Pearson. logtrained_raw = current model, preds back-transformed;', flush=True)
        print('rawtrained_raw = model re-fit on raw targets. RMSE columns are in each endpoint own unit.\n', flush=True)
        print(compare_winners().to_string(index=False), flush=True)
