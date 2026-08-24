"""Build custom Novartis-column multitask datasets for the temporal-fraction chemprop clusters (env `ML`).

For each cluster in config TEMPORAL_FRACTIONS.clusters, writes tf_novartis_<name>.parquet =
[smiles, <safe aux-task columns>, _ik] — the requested raw Novartis pred(...) columns as auxiliary
co-training tasks. Reuses the canonical Novartis SMILES + cached InChIKeys from the already-built
public_novartis_<target> parquet (so leak-removal in the driver is a cheap _ik lookup); labels come
from the raw NOVARTIS_CSV, joined on SMILES. Aggregate stdout only (no SMILES printed). Run once:
  ~/miniconda3/envs/ML/bin/python python/build_novartis_clusters.py
"""
import re, sys, os
from pathlib import Path
import pandas as pd, yaml
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
IKCACHE = CACHE / '_ikcache'
CONFIG = ROOT / 'config/config.yaml'
sys.path.insert(0, str(CACHE))              # build_inputs.compute_descriptastorus (run in `chemprop` env)


def safe(col):   # 'pred(LE-MDCKv2_LogPapp)' -> 'LE_MDCKv2_LogPapp'
    return re.sub(r'[^0-9a-zA-Z]+', '_', col.replace('pred(', '').replace(')', '')).strip('_')


def build_cluster(name, target, novartis_cols, raw_csv):
    canon = pd.read_parquet(CACHE / f'public_novartis_{target}.parquet', columns=['compound', 'smiles'])
    ikc = pd.read_parquet(IKCACHE / f'public_novartis_{target}.parquet.parquet')          # compound,_ik
    canon = canon.merge(ikc, on='compound', how='left').drop_duplicates('smiles')          # canonical smiles + _ik
    raw = pd.read_csv(raw_csv, usecols=['smiles'] + novartis_cols, low_memory=False).drop_duplicates('smiles')
    raw = raw.rename(columns={c: safe(c) for c in novartis_cols})
    out = canon.merge(raw, on='smiles', how='inner')[['smiles', '_ik'] + [safe(c) for c in novartis_cols]]
    fp = CACHE / f'tf_novartis_{name}.parquet'; out.to_parquet(fp, index=False)
    filled = {safe(c): int(out[safe(c)].notna().sum()) for c in novartis_cols}
    print(f'{name}: target={target} rows={len(out)} -> {fp.name}\n  aux-task non-null counts: {filled}', flush=True)


def build_ds_cache():
    """Global smiles -> 200 RDKit2DNormalized (DS_*) lookup so chemprop LOADS descriptors (--descriptors-columns)
    instead of recomputing descriptastorus for 273K rows each run. Public DS reused from the parquets (identical to
    chemprop's v1_rdkit_2d_normalized); internal DS computed once (~325 mols)."""
    from glob import glob
    ds_cols = [c for c in pd.read_parquet(CACHE / 'public_novartis_mdck.parquet', columns=None).columns if c.startswith('DS_')]
    frames = []
    for f in [CACHE / 'public_novartis_mdck.parquet', CACHE / 'public_admetlab_mdck.parquet'] + \
             [Path(p) for p in glob(str(CACHE / 'public_*.parquet')) if 'novartis' not in p and 'admetlab' not in p]:
        if f.exists():
            frames.append(pd.read_parquet(f, columns=['smiles'] + ds_cols))
    from build_inputs import compute_descriptastorus                              # internal DS (small; not stored)
    internal = pd.read_parquet(CACHE / 'internal_targets.parquet', columns=['compound', 'smiles'])
    ids = compute_descriptastorus(internal[['compound', 'smiles']]).drop(columns=['compound'])
    frames.append(pd.concat([internal[['smiles']].reset_index(drop=True), ids.reset_index(drop=True)], axis=1))
    cache = pd.concat(frames, ignore_index=True).drop_duplicates('smiles').reset_index(drop=True)
    cache[ds_cols] = cache[ds_cols].fillna(0.0)                                   # descriptastorus NaN -> 0 (matches failed-mol handling)
    fp = CACHE / 'tf_ds_cache.parquet'; cache.to_parquet(fp, index=False)
    print(f'ds_cache: {len(cache)} unique smiles x {len(ds_cols)} DS cols -> {fp.name}', flush=True)


def main():
    cfg = yaml.safe_load(CONFIG.read_text()); tf = cfg['TEMPORAL_FRACTIONS']; raw = ROOT / cfg['NOVARTIS_CSV']
    for name, spec in tf['clusters'].items():
        build_cluster(name, spec['target'], spec['novartis_cols'], raw)
    build_ds_cache()


if __name__ == '__main__':
    main()
