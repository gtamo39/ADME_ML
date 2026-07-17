"""Extract in-house ADME endpoints from the CDD export into a tidy modelling table.

Reads the endpoint spec from config/config.yaml (`ADME_ENDPOINTS`): each endpoint maps
to one raw assay column + a transform into modelling space. Produces a WIDE table
`[compound, smiles, <endpoint> ...]` where each endpoint column is the transformed target
(NaN where unmeasured) — the natural shape for a masked multitask model, and trivially
subset per endpoint for single-task. Nothing here prints SMILES or per-compound values.

Transforms:
    log10      : log10(x), rows with x<=0 dropped for that endpoint (assay value must be >0)
    identity   : x as-is (already a log-scale readout, e.g. LogD7.4)
    logit_pct  : x is a percentage in (0,100); f = clip(x/100, eps, 1-eps); log10(f/(1-f))
"""
import numpy as np, pandas as pd, yaml

EPS = 1e-3


def _transform(s, kind):
    if kind == 'identity':
        return s.astype(float)
    if kind == 'log10':
        v = s.astype(float)
        return np.where(v > 0, np.log10(v.where(v > 0)), np.nan)
    if kind == 'logit_pct':
        f = (s.astype(float) / 100).clip(EPS, 1 - EPS)
        return np.log10(f / (1 - f))
    raise ValueError(f'unknown transform: {kind}')


def load_adme_spec(config_path='config/config.yaml'):
    """Return (endpoints_dict, id_col, smiles_col, csv_path) from the YAML config."""
    cfg = yaml.safe_load(open(config_path))
    return (cfg['ADME_ENDPOINTS'], cfg['ADME_ID_COL'],
            cfg['ADME_SMILES_COL'], cfg['ADME_CSV'])


def extract_adme_targets(config_path='config/config.yaml', csv_path=None, v=True):
    """Build the wide in-house target table.

    :param config_path: YAML holding ADME_ENDPOINTS / ADME_ID_COL / ADME_SMILES_COL / ADME_CSV
    :param csv_path: override the CSV path from config
    :param v: print per-endpoint non-null counts (aggregate only, no values)
    :return: DataFrame[compound, smiles, <endpoint> ...] in modelling space
    """
    endpoints, id_col, smi_col, cfg_csv = load_adme_spec(config_path)
    df = pd.read_csv(csv_path or cfg_csv, low_memory=False)

    out = pd.DataFrame({'compound': df[id_col].astype(str), 'smiles': df[smi_col].astype(str)})
    for ep, spec in endpoints.items():
        vals = _transform(df[spec['col']], spec['transform'])
        if 'filter' in spec:  # keep only rows whose filter col contains the pattern; null the rest
            keep = df[spec['filter']['col']].astype(str).str.contains(spec['filter']['contains'], na=False)
            vals = np.where(keep, vals, np.nan)
        out[ep] = vals
    out = out.drop_duplicates('smiles').reset_index(drop=True)

    if v:
        print(f'in-house ADME table: {out.shape[0]} compounds x {len(endpoints)} endpoints')
        for ep in endpoints:
            print(f'  {ep:11} n={int(out[ep].notna().sum()):4}  '
                  f'[{endpoints[ep]["transform"]} of {endpoints[ep]["unit"]}]')
    return out


if __name__ == '__main__':
    extract_adme_targets()
