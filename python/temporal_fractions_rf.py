"""RF single-task temporal fraction-split evaluation (env `ML`).

Three single-cut temporal splits per endpoint (train oldest 1-f, test newest f, f in TEMPORAL_FRACTIONS;
ordered by SRB id). Per endpoint: an `internal` arm (internal train only) + one `augmented` arm per config
source-set (internal train + FULL public, NO cap). Test compounds are the newest-f internal, held out of
training; any public row whose InChIKey matches a test compound is dropped ("virtual enumerated" holdout).
Every pred_df records compound/real_y/pred_y (MODELLING space, like the other runs) + applicability-domain
columns (nn_tanimoto_dist, scaffold_novel). Aggregate stdout only.

Run:  ~/miniconda3/envs/ML/bin/python python/temporal_fractions_rf.py [--endpoints hlm,rlm]
  -> output/predictions_runs_temporal_fractions/<ep>/f{40,30,20}/rf_{internal,<sources>}.parquet
"""
import argparse, sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.dirname(__file__))
from adme_cleaning_sweep import ROOT, CACHE, CONFIG, load, _feats
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

NO_CAP = 10 ** 9   # load() subsamples predicted sources only if len>cap; this keeps them FULL


def _fp(smi):
    m = Chem.MolFromSmiles(str(smi));  return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None
def _scaffold(smi):
    m = Chem.MolFromSmiles(str(smi));  return (MurckoScaffold.MurckoScaffoldSmiles(mol=m) or None) if m else None


def add_ad_columns(test, train_smiles):
    """nn_tanimoto_dist (1 - max ECFP4 Tanimoto to train) + scaffold_novel (Murcko not in train)."""
    tr_fps = [f for f in map(_fp, train_smiles) if f is not None]
    tr_scaf = {s for s in map(_scaffold, train_smiles) if s}
    dist, novel = [], []
    for smi in test['smiles']:
        f = _fp(smi)
        dist.append(1.0 - max(DataStructs.BulkTanimotoSimilarity(f, tr_fps)) if (f and tr_fps) else np.nan)
        s = _scaffold(smi); novel.append(bool(s and s not in tr_scaf))
    out = test.copy(); out['nn_tanimoto_dist'] = dist; out['scaffold_novel'] = novel
    return out


def _r2det(y, p):
    ss = ((y - y.mean()) ** 2).sum();  return float('nan') if ss == 0 else 1.0 - ((y - p) ** 2).sum() / ss


def run_endpoint(ep, spec, mf, feats, fractions, rf_kw, out):
    internal, pub, _ = load(ep, mf, feats, NO_CAP, rf_kw['random_state'])
    srb = internal['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
    d = internal.iloc[np.argsort(srb)].reset_index(drop=True); n = len(d)      # oldest -> newest by SRB id
    print(f'\n=== {ep} | internal={n} | public_by_source={ {s: int((pub.source == s).sum()) for s in pub.source.unique()} } ===', flush=True)
    for f in fractions:
        cut = int(n * (1 - f)); tr_int, te = d.iloc[:cut], d.iloc[cut:]
        if len(te) < 3 or len(tr_int) < 3:
            print(f'  f={f}: too few rows (train {len(tr_int)}, test {len(te)}) — skip', flush=True); continue
        test_iks = set(te['_ik'].dropna())
        arms = {'internal': tr_int}
        for arm in spec['augmented']:
            srcs = arm.split('+')
            pubc = pub[pub.source.isin(srcs) & ~pub['_ik'].isin(test_iks)]     # arm sources, drop test-leak
            arms[arm.replace('+', '_')] = pd.concat([tr_int, pubc], ignore_index=True)
        odir = out / ep / f'f{int(round(f * 100))}'; odir.mkdir(parents=True, exist_ok=True)
        for label, train in arms.items():
            rf = RandomForestRegressor(**rf_kw).fit(train[feats], train['label'])
            pred = add_ad_columns(te[['compound', 'smiles']].assign(real_y=te['label'].to_numpy(),
                                                                    pred_y=rf.predict(te[feats])), tr_int['smiles'])   # AD vs INTERNAL train
            pred[['compound', 'real_y', 'pred_y', 'nn_tanimoto_dist', 'scaffold_novel']].to_parquet(odir / f'rf_{label}.parquet', index=False)
            y, p = pred['real_y'].to_numpy(), pred['pred_y'].to_numpy()
            print(f'  f={f} test_n={len(te)} std={y.std():.2f} [{label:11}] n_train={len(train):6} '
                  f'R2det={_r2det(y, p):.3f} RMSE={np.sqrt(((y - p) ** 2).mean()):.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', default=None, help='comma-separated subset (default = all in config)')
    ap.add_argument('--fractions', default=None, help='comma floats override (default = config)')
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text()); tf = cfg['TEMPORAL_FRACTIONS']; rf = cfg['RF_SINGLETASK']
    rf_kw = dict(rf['champion'], n_jobs=rf['n_jobs'], random_state=rf['seed'])
    mf, feats = _feats(); out = ROOT / tf['output_dir']
    eps = args.endpoints.split(',') if args.endpoints else list(tf['endpoints'])
    fracs = [float(x) for x in args.fractions.split(',')] if args.fractions else tf['fractions']
    print(f'> RF temporal fractions {fracs} | {len(feats)} feats | endpoints={eps}', flush=True)
    for ep in eps:
        run_endpoint(ep, tf['endpoints'][ep], mf, feats, fracs, rf_kw, out)
    print(f'\n> done -> {out}', flush=True)


if __name__ == '__main__':
    main()
