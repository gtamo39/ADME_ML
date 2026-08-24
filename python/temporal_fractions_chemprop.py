"""Chemprop temporal fraction-split eval (env `chemprop`, GPU) — grouping arms + custom Novartis-column clusters.

Per config TEMPORAL_FRACTIONS: for each chemprop endpoint, trains ONE multitask model per arm and scores only
that endpoint on its newest-f temporal test (held out ENTIRELY across all tasks — "virtual enumerated set"):
  - GROUPING arms (cp_arms on cp `chemprop` grouping): internal wide + EXP(target, single-endpoint) + WIDE
    NVS/ADM (all covered tasks, FULL, no cap).
  - CLUSTER arms (cp_clusters): internal `target` + the cluster's Novartis pred columns as auxiliary tasks.
Descriptors: precomputed RDKit2DNormalized (DS_*) loaded from tf_ds_cache.parquet via --descriptors-columns
(identical to v1_rdkit_2d_normalized but computed ONCE — no per-run descriptastorus). Fast HP + epochs from
config, ensemble=1. Public rows whose InChIKey matches a test compound are dropped. Saves pred_dfs
(compound/real_y/pred_y MODELLING space + nn_tanimoto_dist/scaffold_novel vs INTERNAL train).

Run:  ~/miniconda3/envs/chemprop/bin/python python/temporal_fractions_chemprop.py [--endpoints hlm,mdck] [--fast]
"""
import argparse, sys, os
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.dirname(__file__))
import run_Chemprop_SystematicGroups as G
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

CACHE, PUBLIC_SRC, ROOT, cp = G.CACHE, G.PUBLIC_SRC, G.ROOT, G.cp
IKCACHE = CACHE / '_ikcache'
RUN = CACHE / 'chemprop_run'


def _fp(smi):
    m = Chem.MolFromSmiles(str(smi));  return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None
def _scaffold(smi):
    m = Chem.MolFromSmiles(str(smi));  return (MurckoScaffold.MurckoScaffoldSmiles(mol=m) or None) if m else None
def add_ad_columns(test, train_smiles):
    tr_fps = [f for f in map(_fp, train_smiles) if f is not None]; tr_scaf = {s for s in map(_scaffold, train_smiles) if s}
    dist, novel = [], []
    for smi in test['smiles']:
        f = _fp(smi); dist.append(1.0 - max(DataStructs.BulkTanimotoSimilarity(f, tr_fps)) if (f and tr_fps) else np.nan)
        s = _scaffold(smi); novel.append(bool(s and s not in tr_scaf))
    out = test.copy(); out['nn_tanimoto_dist'] = dist; out['scaffold_novel'] = novel
    return out


def newest_frac_ids(tgt, ep, f):
    d = tgt.dropna(subset=[ep]); srb = d['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
    return set(d['compound'].to_numpy()[np.argsort(srb)[int(len(d) * (1 - f)):]])
def iks_of(smiles):
    out = set()
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m and (k := Chem.MolToInchiKey(m)):
            out.add(k)
    return out


def _drop_leak(frame, cache_tag, exclude_iks):
    ikf = IKCACHE / f'{cache_tag}.parquet'
    if not ikf.exists() or not exclude_iks:
        return frame.drop(columns=['compound'])
    frame = frame.merge(pd.read_parquet(ikf), on='compound', how='left')
    return frame[~frame['_ik'].isin(exclude_iks)].drop(columns=['compound', '_ik'])

def exp_rows(ep, exclude_iks):
    p = CACHE / PUBLIC_SRC['EXP'].format(ep=ep)
    if not p.exists():
        return None
    r = pd.read_parquet(p, columns=['compound', 'smiles', 'value']).rename(columns={'value': ep})
    return _drop_leak(r, PUBLIC_SRC['EXP'].format(ep=ep), exclude_iks)

def wide_public(key, grp_eps, exclude_iks):
    present = [ep for ep in grp_eps if (CACHE / PUBLIC_SRC[key].format(ep=ep)).exists()]
    if not present:
        return None
    wide = None
    for ep in present:
        f = pd.read_parquet(CACHE / PUBLIC_SRC[key].format(ep=ep), columns=['compound', 'smiles', 'value']).rename(columns={'value': ep})
        wide = f if wide is None else wide.merge(f[['compound', ep]], on='compound', how='outer')
    return _drop_leak(wide, PUBLIC_SRC[key].format(ep=present[0]), exclude_iks)


def _splits(tgt, test_ids, rng):
    s = np.where(tgt['compound'].isin(test_ids), 'test', 'train').astype(object)
    tr = np.flatnonzero(s == 'train'); s[rng.choice(tr, size=max(1, int(len(tr) * 0.1)), replace=False)] = 'val'
    return s


def build_grouping(tgt, grp_eps, target, f, srcs, rng):
    test_ids = newest_frac_ids(tgt, target, f)
    test_truth = tgt[tgt['compound'].isin(test_ids)][['compound', 'smiles'] + grp_eps].reset_index(drop=True)
    excl = iks_of(test_truth['smiles'])
    internal = tgt[['smiles'] + grp_eps].copy(); internal['splits'] = _splits(tgt, test_ids, rng)
    rows = [internal[['smiles', 'splits'] + grp_eps]]
    if 'EXP' in srcs and (r := exp_rows(target, excl)) is not None:
        r['splits'] = 'train'; rows.append(r)
    for key in ('NVS', 'ADM'):
        if key in srcs and (w := wide_public(key, grp_eps, excl)) is not None:
            w['splits'] = 'train'; rows.append(w)
    combined = pd.concat(rows, ignore_index=True)
    for ep in grp_eps:
        if ep not in combined:
            combined[ep] = np.nan
    return combined[['smiles', 'splits'] + grp_eps], test_truth, grp_eps


def build_cluster(tgt, target, cl, f, rng):
    aux = pd.read_parquet(CACHE / f'tf_novartis_{cl}.parquet')
    aux_names = [c for c in aux.columns if c not in ('smiles', '_ik')]; tasks = [target] + aux_names
    test_ids = newest_frac_ids(tgt, target, f)
    test_truth = tgt[tgt['compound'].isin(test_ids)][['compound', 'smiles', target]].reset_index(drop=True)
    excl = iks_of(test_truth['smiles'])
    internal = tgt[['smiles', target]].copy(); internal['splits'] = _splits(tgt, test_ids, rng)
    for a in aux_names:
        internal[a] = np.nan
    pub = aux[~aux['_ik'].isin(excl)].drop(columns=['_ik']).copy(); pub['splits'] = 'train'; pub[target] = np.nan
    combined = pd.concat([internal[['smiles', 'splits'] + tasks], pub[['smiles', 'splits'] + tasks]], ignore_index=True)
    return combined, test_truth, tasks


def attach_ds(frame, ds, ds_cols):
    out = frame.merge(ds, on='smiles', how='left')
    out[ds_cols] = out[ds_cols].fillna(0.0)
    return out


def train_predict_ds(params, gname, tasks, combined_path, test_path, ds_cols):
    model_dir = RUN / f'tf_model_{gname}'; hp = params.hp
    cmd = [cp.CHEMPROP, 'train', '-i', str(combined_path), '-s', 'smiles', '--target-columns', *tasks,
           '--splits-column', 'splits', '-t', 'regression', '--metrics', 'rmse', 'mae',
           '--epochs', str(params.epochs), '--patience', str(params.patience), '--num-workers', '0',
           '-o', str(model_dir), '--data-seed', str(params.seed), '--descriptors-columns', *ds_cols]
    for flag, key in [('--depth', 'depth'), ('--message-hidden-dim', 'message_hidden_dim'), ('--ffn-num-layers', 'ffn_num_layers'),
                      ('--ffn-hidden-dim', 'ffn_hidden_dim'), ('--dropout', 'dropout'), ('-b', 'batch_size'), ('--aggregation', 'aggregation')]:
        if key in hp:
            cmd += [flag, str(hp[key])]
    cp._run_quiet(cmd, RUN / f'tf_train_{gname}.log')
    ckpt = max(list(model_dir.rglob('best*.ckpt')) or list(model_dir.rglob('*.ckpt')), key=lambda p: p.stat().st_mtime)
    preds = RUN / f'tf_preds_{gname}.csv'
    cp._run_quiet([cp.CHEMPROP, 'predict', '-i', str(test_path), '-s', 'smiles', '--model-path', str(ckpt),
                   '--preds-path', str(preds), '--descriptors-columns', *ds_cols], RUN / f'tf_predict_{gname}.log')
    return pd.read_csv(preds)


def run_arm(ep, target, tasks, combined, test_truth, label, params, tf, tgt, f, ds, ds_cols):
    RUN.mkdir(exist_ok=True); gname = f'{ep}_f{int(round(f * 100))}_{label}'
    cpath = RUN / f'tf_combined_{gname}.csv'; attach_ds(combined, ds, ds_cols).to_csv(cpath, index=False)
    tpath = RUN / f'tf_test_{gname}.csv'; attach_ds(test_truth[['smiles']], ds, ds_cols).to_csv(tpath, index=False)
    print(f'\n[{gname}] tasks={tasks} train={int((combined.splits == "train").sum())} test={len(test_truth)}', flush=True)
    preds = train_predict_ds(params, gname, tasks, cpath, tpath, ds_cols)
    col = f'pred_{tasks.index(target)}'; col = col if col in preds.columns else target
    df = pd.DataFrame({'compound': test_truth['compound'], 'smiles': test_truth['smiles'],
                       'real_y': test_truth[target].to_numpy(float), 'pred_y': preds[col].to_numpy(float)}).dropna(subset=['real_y', 'pred_y']).reset_index(drop=True)
    tr_smiles = tgt.loc[~tgt['compound'].isin(newest_frac_ids(tgt, target, f)) & tgt[target].notna(), 'smiles']
    df = add_ad_columns(df, tr_smiles)
    d = ROOT / tf['output_dir'] / ep / f'f{int(round(f * 100))}'; d.mkdir(parents=True, exist_ok=True)
    df[['compound', 'real_y', 'pred_y', 'nn_tanimoto_dist', 'scaffold_novel']].to_parquet(d / f'cp_{label}.parquet', index=False)
    y, p = df['real_y'].to_numpy(), df['pred_y'].to_numpy()
    r2d = float('nan') if y.std() == 0 else 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'  -> {ep} [{label}] n={len(df)} std={y.std():.2f} R2det={r2d:.3f} RMSE={np.sqrt(((y - p) ** 2).mean()):.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', default=None); ap.add_argument('--fast', action='store_true')
    ap.add_argument('--fractions', default=None, help='comma floats override (default = config)')
    args = ap.parse_args()
    params = G.PARAMS(); data = G.DATA().load_all(params); tgt = data.tgt
    cfg = yaml.safe_load(G.CONFIG.read_text()); tf = cfg['TEMPORAL_FRACTIONS']
    params.hp = tf['chemprop_hp']; params.epochs = tf['chemprop_epochs']            # fast small model, ensemble=1
    if args.fast:
        params.epochs = 5; params.patience = 5
    ds = pd.read_parquet(CACHE / 'tf_ds_cache.parquet'); ds_cols = [c for c in ds.columns if c.startswith('DS_')]
    eps = args.endpoints.split(',') if args.endpoints else [e for e, s in tf['endpoints'].items() if s.get('chemprop') or s.get('cp_clusters')]
    fracs = [float(x) for x in args.fractions.split(',')] if args.fractions else tf['fractions']
    print(f'> chemprop temporal fractions {fracs} | epochs={params.epochs} hp={params.hp} | DS={len(ds_cols)} precomputed | endpoints={eps}', flush=True)
    for ep in eps:
        spec = tf['endpoints'][ep]
        for f in fracs:
            if spec.get('chemprop'):                                               # grouping arms
                grp_eps = params.groupings[spec['chemprop']]
                for arm in spec.get('cp_arms', []):
                    combined, test_truth, tasks = build_grouping(tgt, grp_eps, ep, f, arm.split('+'), np.random.default_rng(params.seed))
                    run_arm(ep, ep, tasks, combined, test_truth, arm.replace('+', '_'), params, tf, tgt, f, ds, ds_cols)
            for cl in spec.get('cp_clusters', []):                                 # custom cluster arms
                combined, test_truth, tasks = build_cluster(tgt, ep, cl, f, np.random.default_rng(params.seed))
                run_arm(ep, ep, tasks, combined, test_truth, cl, params, tf, tgt, f, ds, ds_cols)
    print(f'\n> done -> {ROOT / tf["output_dir"]}', flush=True)


if __name__ == '__main__':
    main()
