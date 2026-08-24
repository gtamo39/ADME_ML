"""Realistic temporal evaluation for the internal ADME endpoints.

The single-cut "newest-30%" split is degenerate: the newest block's target variance collapses
(e.g. rlm newest-12 are 83% floor-censored, std 0.10) so SS_tot≈0 and R²det explodes to −100s (see
wiki "CV is calibrated, temporal is NOT"). This replaces it with three honest read-outs per endpoint:

  1. ROLLING-ORIGIN (expanding-window) temporal CV, DISJOINT next-block test, predictions POOLED.
     Sort internal by SRB id; fold i trains on the oldest ANCHORS[i] + all policy-public and tests the
     NEXT slice only. Every newest-50% compound is scored exactly once by a model trained only on its
     past → pool → headline RMSE / bias / R²det (+ bootstrap CI) on a full-variance set.
  2. FRACTION-SENSITIVITY curve: train oldest (1−f), test newest f (all-remaining) for f in FRACTIONS —
     how accuracy decays the further into the future we hold out.
  3. APPLICABILITY-DOMAIN columns on every pred_df (structure-free, stay local): nn_tanimoto_dist
     (1 − max ECFP4 Tanimoto to train) and scaffold_novel (Bemis-Murcko scaffold absent from train).

Per endpoint it compares arms: internal_only vs the cleaning-sweep-recommended source policy (WINNER).
Aggregate stdout only — no SMILES / per-compound values printed; pred_dfs hold ids + numbers only.

Run (env `ML`):  python python/temporal_eval.py [--endpoints logd,rlm] [--arms internal_only,recommended]
  -> output/predictions_runs_temporal/<ep>/{rolling_<arm>.parquet, curve.csv, temporal_summary.csv}
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output/predictions_runs_temporal'
sys.path.insert(0, os.path.expanduser('~/Scripts'))
sys.path.insert(0, str(ROOT / 'python'))
from run_RF_SingleTask_systematic import PARAMS, DATA           # reuse loaders / champion params
from Statistics_tools import rsquared as rsq                    # "old way" R2 = squared Pearson
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

ANCHORS = [0.5, 0.6, 0.7, 0.8, 0.9]                             # expanding-train edges; test = the next slice
FRACTIONS = [0.4, 0.3, 0.2]                                     # sensitivity curve = newest-f% held out (all-remaining); 0.1 too small for n≤107
CAP = 40000                                                     # subsample predicted sources (NVS 273k) for tractability
# cleaning-sweep CV winners (2026-07-21), for reference/display ([]=internal-only). Temporal RE-TESTS these, not assumes them.
WINNER = {'solubility': ['EXP'], 'logd': [], 'hlm': [], 'mlm': [], 'rlm': ['EXP'], 'mdck': [], 'ppb': ['EXP'], 'caco2': []}


def arms_for(ep, data, params):
    """Arms tested under the temporal protocol (independent of the CV pick, so temporal can overturn it):
    previous_deployed = current config policy (the 'previous' baseline); internal-only ALWAYS; experimental
    ([EXP]) wherever it exists (the key head-to-head); else the deployed predicted source. Duplicate source
    sets are deduped (keep first label)."""
    avail = data.avail_sources(ep)
    cand = [('previous_deployed', list(data.augmented_sources(ep, params))), ('internal_only', [])]
    if 'EXP' in avail:
        cand.append(('experimental', ['EXP']))
    else:
        pred = [s for s in ('NVS', 'ADM') if s in avail][:1]
        if pred:
            cand.append((f'predicted_{pred[0].lower()}', pred))
    seen, arms = set(), []
    for name, src in cand:
        key = tuple(sorted(src))
        if key not in seen:
            seen.add(key); arms.append((name, src))
    return arms


# ----------------------------------------------------------------- applicability domain (structure-free outputs)
def _fp(smi):
    m = Chem.MolFromSmiles(str(smi));  return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None
def _scaffold(smi):
    m = Chem.MolFromSmiles(str(smi));  return (MurckoScaffold.MurckoScaffoldSmiles(mol=m) or None) if m else None


def add_ad_columns(test, train_smiles):
    """Append nn_tanimoto_dist (1 − max ECFP4 Tanimoto to train) and scaffold_novel (Murcko not in train)."""
    tr_fps = [f for f in map(_fp, train_smiles) if f is not None]
    tr_scaf = {s for s in map(_scaffold, train_smiles) if s}
    dist, novel = [], []
    for smi in test['smiles']:
        f = _fp(smi)
        dist.append(1.0 - max(DataStructs.BulkTanimotoSimilarity(f, tr_fps)) if (f and tr_fps) else np.nan)
        s = _scaffold(smi); novel.append(bool(s and s not in tr_scaf))
    test = test.copy(); test['nn_tanimoto_dist'] = dist; test['scaffold_novel'] = novel
    return test


# ----------------------------------------------------------------- metrics
def _r2det(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); ss = ((y - y.mean()) ** 2).sum()
    return float('nan') if ss == 0 else 1.0 - ((y - p) ** 2).sum() / ss
def _rmse(y, p):  return float(np.sqrt(((np.asarray(y, float) - np.asarray(p, float)) ** 2).mean()))


def _boot_ci(y, p, fn, B=1000, seed=42):
    rng = np.random.default_rng(seed); n = len(y); y, p = np.asarray(y, float), np.asarray(p, float)
    vals = [fn(y[idx], p[idx]) for idx in (rng.integers(0, n, n) for _ in range(B))]
    return round(float(np.nanpercentile(vals, 2.5)), 3), round(float(np.nanpercentile(vals, 97.5)), 3)


def _metrics(d, label=''):
    y, p = d['real_y'].to_numpy(), d['pred_y'].to_numpy()
    lo_r, hi_r = _boot_ci(y, p, _rmse); lo_d, hi_d = _boot_ci(y, p, _r2det)
    return {'arm': label, 'n_test': len(d), 'test_std': round(float(np.std(y)), 3),
            'RMSE': round(_rmse(y, p), 4), 'RMSE_CI': f'[{lo_r},{hi_r}]', 'bias': round(float((p - y).mean()), 3),
            'R2det': round(_r2det(y, p), 3), 'R2det_CI': f'[{lo_d},{hi_d}]', 'R2': round(float(rsq(y, p)), 3)}


# ----------------------------------------------------------------- splits + fit
def _srb_order(df):
    return df.iloc[np.argsort(df['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy())]


def _fit_predict(train, test, feats, rf_kw):
    rf = RandomForestRegressor(**rf_kw).fit(train[feats], train['label'])
    out = test[['compound', 'smiles']].copy(); out['real_y'] = test['label'].to_numpy(); out['pred_y'] = rf.predict(test[feats])
    return out


def eval_arm(ep, sources, data, feats, rf_kw):
    """Rolling-origin (pooled, disjoint blocks) + fraction curve for one source policy. Returns (pooled_df, curve_rows)."""
    _, d, pub = data.pooled(ep, sources)
    pubc = pd.concat(pub, ignore_index=True) if pub else d.iloc[0:0]
    if len(pubc) > CAP:                                          # cap predicted sources (NVS 273k) uniformly
        pubc = pubc.sample(n=CAP, random_state=42).reset_index(drop=True)
    do = _srb_order(d); n = len(do)
    # 1. rolling-origin, disjoint next-block test, pooled
    edges = [int(n * a) for a in ANCHORS] + [n]
    blocks = []
    for i in range(len(ANCHORS)):
        tr_int, te = do.iloc[:edges[i]], do.iloc[edges[i]:edges[i + 1]]
        if len(te) == 0 or len(tr_int) < 3:
            continue
        train = pd.concat([tr_int, pubc], ignore_index=True)
        pred = add_ad_columns(_fit_predict(train, te, feats, rf_kw), train['smiles'])
        pred['fold'] = i
        blocks.append(pred)
    pooled = pd.concat(blocks, ignore_index=True) if blocks else d.iloc[0:0]
    # 2. fraction-sensitivity curve (train oldest 1−f, test all-remaining newest f)
    curve = []
    for f in FRACTIONS:
        cut = int(n * (1 - f)); tr_int, te = do.iloc[:cut], do.iloc[cut:]
        if len(te) < 3 or len(tr_int) < 3:
            continue
        train = pd.concat([tr_int, pubc], ignore_index=True)
        pr = _fit_predict(train, te, feats, rf_kw)
        curve.append({'endpoint': ep, 'sources': '+'.join(sources) or 'internal_only', 'test_frac': f,
                      'n_test': len(pr), 'test_std': round(float(pr['real_y'].std()), 3),
                      'RMSE': round(_rmse(pr['real_y'], pr['pred_y']), 4), 'bias': round(float((pr['pred_y'] - pr['real_y']).mean()), 3),
                      'R2det': round(_r2det(pr['real_y'], pr['pred_y']), 3)})
    return pooled, curve


def run(endpoints, data, params):
    feats, rf_kw = data.feats, params.rf_kw
    summ, curves = [], []
    for ep in endpoints:
        print(f'\n=== {ep} (rolling-origin, disjoint pooled newest-50%; CV winner: {"+".join(WINNER[ep]) or "internal-only"}) ===', flush=True)
        for name, sources in arms_for(ep, data, params):
            pooled, curve = eval_arm(ep, sources, data, feats, rf_kw)
            curves += curve
            if len(pooled) < 3:
                print(f'  {name:16} too few test rows', flush=True); continue
            (OUT / ep).mkdir(parents=True, exist_ok=True)
            pooled[['compound', 'real_y', 'pred_y', 'nn_tanimoto_dist', 'scaffold_novel', 'fold']].to_parquet(OUT / ep / f'rolling_{name}.parquet', index=False)
            m = {'endpoint': ep, 'sources': '+'.join(sources) or 'internal_only', **_metrics(pooled, name)}
            summ.append(m)
            print(f"  {name:16} n={m['n_test']:3} test_std={m['test_std']:.2f}  RMSE={m['RMSE']} {m['RMSE_CI']}  "
                  f"bias={m['bias']}  R2det={m['R2det']} {m['R2det_CI']}  R2={m['R2']}  novel_scaffold={int(pooled['scaffold_novel'].sum())}/{len(pooled)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summ).to_csv(OUT / 'temporal_summary.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'temporal_curve.csv', index=False)
    print(f'\n> wrote {OUT}/temporal_summary.csv + temporal_curve.csv', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', default=','.join(WINNER), help='comma-separated subset')
    args = ap.parse_args()
    params = PARAMS(); data = DATA().load_all(params)
    run([e for e in args.endpoints.split(',') if e in WINNER], data, params)


if __name__ == '__main__':
    main()
