"""Shared test fixture: a small (~1K-compound) LOCAL subset of the real cached ADME parquets.

Materialized into a temp dir so the systematic runners can be exercised end-to-end fast, without
touching the production `autoresearch/predict_adme` cache or the real output dirs. Values stay on
disk and are NEVER printed — the tests assert only counts/booleans/metrics. Cached H236 features
already exist in the real cache, so the subset is pure row-slicing (no re-featurization).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_CACHE = ROOT / 'autoresearch/predict_adme'
PUBLIC_EPS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2']   # endpoints given a public-EXP subset
N_PUBLIC = 160                                                      # rows sliced per public parquet (~1K total)


def _first_rows(path, n):
    """First ~n rows of a parquet via a single row-batch — cheap and low-memory (no full-file read)."""
    return next(pq.ParquetFile(path).iter_batches(batch_size=n)).to_pandas()


def build(dest):
    """Materialize the subset cache under `dest`; return its Path. ~325 internal + N_PUBLIC*len(PUBLIC_EPS)
    public compounds, schema-identical to the real cache (public parquets keep the full H236 feature set)."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    tgt = pd.read_parquet(REAL_CACHE / 'internal_targets.parquet')
    tgt.to_parquet(dest / 'internal_targets.parquet', index=False)
    mf = pd.read_parquet(REAL_CACHE / 'internal_MF.parquet').drop_duplicates('compound')
    mf[mf['compound'].isin(tgt['compound'])].to_parquet(dest / 'internal_MF.parquet', index=False)
    for ep in PUBLIC_EPS:
        src = REAL_CACHE / f'public_{ep}.parquet'
        if src.exists():
            _first_rows(src, N_PUBLIC).to_parquet(dest / f'public_{ep}.parquet', index=False)
    return dest


def isolated_registry(tmp):
    """A fresh MLTrail Registry backed by an isolated temp vault (built-in featurizers -> H236 works
    without the external Rdkit_tools path)."""
    from mltrail import Registry
    cfg = {'registry_path': str(Path(tmp) / 'reg/registry.json'),
           'trained_models_dir': str(Path(tmp) / 'reg/models'),
           'training_sets_dir': str(Path(tmp) / 'reg/tsets'),
           'date_format': '%Y%m%d_%H%M%S', 'featurizers': {},
           'chemprop': {'cli': 'chemprop', 'accelerator': 'cpu'}}
    return Registry.from_config(cfg)


def fake_chemprop_cli(n_targets):
    """A `subprocess.run` stand-in for `chemprop predict`: reads the `-i` SMILES CSV and writes
    `--preds-path` with `n_targets` deterministic columns (row-index based), preserving row order."""
    import subprocess

    def run(cmd, capture_output=True, text=True):
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
        smi = pd.read_csv(args['-i'])['smiles']
        out = pd.DataFrame({'smiles': smi.values})
        for j in range(n_targets):
            out[f't{j}'] = [float(i + j) for i in range(len(smi))]
        out.to_csv(args['--preds-path'], index=False)
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
    return run
