"""Combine the public-only -> internal results into the final winner table.

Reads the two public-only summaries (RF: summary_public_only.csv; Chemprop:
summary_chemprop_publiconly.csv) and, per endpoint, reports the WINNING model's public-only r2
(config PUBLIC_TO_INTERNAL.winners) next to its with-internal temporal r2 for context — so the
gap = how much our internal training data adds on top of public alone.

Env-agnostic (reads CSVs only). Run:  python python/combine_public_to_internal.py
"""
from pathlib import Path
import pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'config/config.yaml'
RF_DIR = ROOT / 'output/predictions_runs'
CP_DIR = ROOT / 'output/predictions_runs_chemprop'
ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']


def _read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    winners = yaml.safe_load(CONFIG.read_text())['PUBLIC_TO_INTERNAL']['winners']
    rf_pub = _read(RF_DIR / 'summary_public_only.csv').set_index('endpoint') if (RF_DIR / 'summary_public_only.csv').exists() else pd.DataFrame()
    cp_pub = _read(CP_DIR / 'summary_chemprop_publiconly.csv')
    rf_dep = _read(RF_DIR / 'summary.csv').set_index('endpoint') if (RF_DIR / 'summary.csv').exists() else pd.DataFrame()
    cp_dep = _read(CP_DIR / 'summary_chemprop.csv')
    metrics = _read(RF_DIR / 'metrics_all.csv')                  # per-arm RMSE (public-only arms)
    rmse_of = lambda src, ep, arm: (lambda s: float(s.iloc[0]) if len(s) else None)(
        metrics.loc[(metrics.source == src) & (metrics.endpoint == ep) & (metrics.arm == arm), 'rmse'])

    rows = []
    for ep in ENDPOINTS:
        w = winners[ep]; model = w['model']; grp = w.get('grouping')
        if model == 'chemprop':
            pub = cp_pub[(cp_pub['endpoint'] == ep) & (cp_pub['grouping'] == grp)]
            dep = cp_dep[(cp_dep['endpoint'] == ep) & (cp_dep['grouping'] == grp)]
            pub_r2 = pub['publiconly_r2'].iloc[0] if len(pub) else None
            pub_n = int(pub['publiconly_n'].iloc[0]) if len(pub) else 0
            dep_r2 = dep['r2'].iloc[0] if len(dep) else None
            pub_rmse = rmse_of('chemprop', ep, f'{grp}_publiconly')
            label = f'chemprop:{grp}'
        else:
            pub_r2 = rf_pub.loc[ep, 'publiconly_r2'] if ep in rf_pub.index else None
            pub_n = int(rf_pub.loc[ep, 'publiconly_n']) if ep in rf_pub.index else 0
            dep_r2 = rf_dep.loc[ep, 'augmented_temporal_r2'] if ep in rf_dep.index else None
            pub_rmse = rmse_of('RF', ep, 'public_only')
            label = 'RF'
        rows.append({'endpoint': ep, 'winner': label, 'publiconly_r2': pub_r2, 'publiconly_rmse': pub_rmse,
                     'publiconly_n': pub_n, 'deploy_temporal_r2': dep_r2,
                     'gap_from_internal': (None if pub_r2 is None or dep_r2 is None else round(float(dep_r2) - float(pub_r2), 3))})

    out = pd.DataFrame(rows)
    dest = RF_DIR / 'summary_public_to_internal.csv'; out.to_csv(dest, index=False)
    print(f'> wrote {dest}\n')
    print('=== PUBLIC-ONLY -> INTERNAL (winner per endpoint) ===')
    print('publiconly_r2 = trained on public only, predicting ALL internal; '
          'deploy_temporal_r2 = same winner WITH internal in train (context)\n')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
