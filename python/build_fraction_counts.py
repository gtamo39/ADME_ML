"""Provenance (source + n_train) for the temporal fraction-split + renamed rolling arms in Summary_results.ipynb.

Writes output/train_counts_fractions.csv keyed by the notebook pred_df keys:
  - temporalRolling_<arm>       : remapped from train_counts.csv (rolling-origin rows), source+n_train copied.
  - temporal{T}_{F}_{rf|cp}_<arm>: n_train parsed EXACTLY from the run logs (both print it); source derived.
Merged into endpoint_metrics_table's _CNT so those rows show source/n_train instead of '?'/nan. Re-run after
more fractions finish. Run (env ML):  ~/miniconda3/envs/ML/bin/python python/build_fraction_counts.py
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'
EXP_PROV = {'solubility': 'PharmaBench/ChEMBL/Biogen(+)', 'logd': 'AstraZeneca', 'hlm': 'Biogen/AstraZeneca',
            'rlm': 'Biogen', 'caco2': 'Wang(TDC)', 'ppb': 'AstraZeneca/Biogen'}


def derive_source(ep, arm):
    if 'perm' in arm:  return 'Novartis perm-cluster'
    if 'fu' in arm:    return 'Novartis fu-cluster'
    parts = ([EXP_PROV.get(ep, 'experimental')] if 'EXP' in arm else []) + \
            (['Novartis'] if 'NVS' in arm else []) + (['ADMETlab'] if 'ADM' in arm else [])
    return ' + '.join(parts) if parts else 'internal only'


def remap_rolling(key):
    if key == 'EXP_thermo_temporal':    return 'temporalRolling_EXP_thermo'
    if key == 'internal_only_temporal': return 'temporalRolling_internal'
    if key.startswith('temporal_'):     return 'temporalRolling_' + key[len('temporal_'):]
    return None


def parse_logs():
    rows = []
    for lg in sorted(OUT.glob('temporal_fractions_rf_run*.log')):       # RF: '=== ep |' headers + 'f=.. [arm] n_train='
        ep = None
        for line in lg.read_text().splitlines():
            if (m := re.match(r'=== (\w+) \| internal=', line)):
                ep = m.group(1)
            elif ep and (m := re.search(r'f=([\d.]+) test_n=\d+ std=[\d.]+ \[(\w+)\s*\] n_train=\s*(\d+)', line)):
                f = int(round(float(m.group(1)) * 100)); arm = m.group(2)
                rows.append((ep, f'temporal{100 - f}_{f}_rf_{arm}', derive_source(ep, arm), int(m.group(3))))
    for lg in sorted(OUT.glob('temporal_fractions_chemprop_run*.log')):  # CP: '[ep_fPCT_arm] .. train=N'
        for line in lg.read_text().splitlines():
            if (m := re.match(r'\[([a-z0-9]+)_f(\d+)_(.+?)\] .*train=(\d+)', line)):
                ep, f, arm = m.group(1), int(m.group(2)), m.group(3)
                rows.append((ep, f'temporal{100 - f}_{f}_cp_{arm}', derive_source(ep, arm), int(m.group(4))))
    return rows


def main():
    out = []
    tc = OUT / 'predictions_runs_temporal_fractions'  # (unused dir; counts come from logs + train_counts.csv)
    src = pd.read_csv(OUT / 'train_counts.csv')                          # rolling rows -> temporalRolling_ keys
    for r in src.itertuples():
        if (nk := remap_rolling(r.key)):
            out.append((r.endpoint, nk, r.providers, int(r.n_train)))
    out += parse_logs()
    df = pd.DataFrame(out, columns=['endpoint', 'key', 'source', 'n_train']).drop_duplicates(['endpoint', 'key'])
    df.to_csv(OUT / 'train_counts_fractions.csv', index=False)
    print(f'> wrote train_counts_fractions.csv ({len(df)} rows)\n{df.to_string(index=False)}', flush=True)


if __name__ == '__main__':
    main()
