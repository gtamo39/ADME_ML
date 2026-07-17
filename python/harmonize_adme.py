"""Harmonize public ADME datasets into the in-house modelling space, per endpoint.

Each loader returns a tidy frame [smiles, value, origin, endpoint] where `value` is in the
SAME modelling space as `extract_adme` (log10 for clearance/permeability, identity for LogD,
log10-logit for PPB fu) — so a public frame can be pooled directly with the in-house table.

Source→unit conversions are driven by config (`ADME_PUBLIC`, `ADME_SCALING`); see wiki
"Public-data → in-house unit mapping". Nothing here prints SMILES or per-row values —
only aggregate shapes/counts. Public CSVs' `smiles` columns are passed through, never echoed.

TDC sets download once via PyTDC (public data only) and cache under `data/public_adme/`.
Biogen-Fang is read from the local CSV.
"""
import os, numpy as np, pandas as pd, yaml
from extract_adme import EPS  # shared logit epsilon -> identical to in-house transform

TDC_CACHE = 'data/public_adme'


def _logit_fu(fu):
    """log10-logit of fraction-unbound; matches extract_adme.logit_pct."""
    f = np.asarray(fu, float).clip(EPS, 1 - EPS)
    return np.log10(f / (1 - f))


def _frame(smiles, value, origin, endpoint):
    out = pd.DataFrame({'smiles': pd.Series(smiles).astype(str), 'value': np.asarray(value, float),
                        'origin': origin, 'endpoint': endpoint})
    return out[out['value'].notna() & (out['smiles'].str.len() > 0)].reset_index(drop=True)


def _load_config(path):
    cfg = yaml.safe_load(open(path))
    return cfg['ADME_PUBLIC'], cfg['ADME_SCALING'], cfg.get('BIOGEN_ADME_CSV')


def _tdc_raw(name):
    """Download/cache a TDC ADMET regression set -> DataFrame[Drug(SMILES), Y]."""
    from tdc.single_pred import ADME
    os.makedirs(TDC_CACHE, exist_ok=True)
    return ADME(name=name, path=TDC_CACHE).get_data()


def _apply_op(y, spec, scaling):
    """Convert a public column `y` to in-house modelling space per its config `op`."""
    op = spec['op']
    if op == 'identity':
        return np.asarray(y, float)
    if op == 'log10':
        v = np.asarray(y, float)
        return np.where(v > 0, np.log10(np.where(v > 0, v, np.nan)), np.nan)
    if op == 'log10_offset':
        return np.asarray(y, float) + spec['offset']
    if op == 'pctbound_to_logit_fu':
        return _logit_fu((100 - np.asarray(y, float)) / 100)
    if op == 'log_pctunbound_to_logit_fu':
        return _logit_fu(np.power(10.0, np.asarray(y, float)) / 100)
    if op == 'log_clint_bw_to_invitro':
        s = scaling[spec['species']]
        # col = log10(mL/min/kg); subtract log10(MPPGL*LW/1000) to reach log10(uL/min/mg)
        return np.asarray(y, float) - np.log10(s['MPPGL'] * s['LW'] / 1000)
    raise ValueError(f'unknown op: {op}')


def harmonize_endpoint(endpoint, config_path='config/config.yaml', v=True):
    """Return the harmonized public frame(s) for one in-house endpoint (pooled across sources).

    :param endpoint: one of the in-house endpoint keys (logd/hlm/rlm/caco2/ppb/...)
    :return: DataFrame[smiles, value, origin, endpoint] in modelling space, or empty if no source
    """
    public, scaling, biogen_csv = _load_config(config_path)
    keys = [k for k in public if k == endpoint or k.startswith(endpoint + '_')]
    frames = []
    for k in keys:
        spec = public[k]
        if spec['source'] == 'tdc':
            raw = _tdc_raw(spec['name'])
            frames.append(_frame(raw['Drug'], _apply_op(raw['Y'], spec, scaling),
                                 f'TDC:{spec["name"]}', endpoint))
        elif spec['source'] == 'biogen':
            raw = pd.read_csv(biogen_csv)
            col = raw[spec['col']]
            frames.append(_frame(raw['SMILES'], _apply_op(col, spec, scaling),
                                 f'Biogen:{k}', endpoint))
    out = pd.concat(frames, ignore_index=True) if frames else _frame([], [], '', endpoint)
    if v:
        by = out.groupby('origin').size().to_dict() if len(out) else {}
        print(f'  {endpoint:11} public n={len(out):5}  {by}')
    return out


def harmonize_all(config_path='config/config.yaml', out_dir='autoresearch/predict_adme', v=True):
    """Harmonize every endpoint that has a public source; cache to parquet. Returns dict[endpoint]->frame."""
    public, _, _ = _load_config(config_path)
    endpoints = sorted({k.split('_')[0] for k in public})
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    if v: print('=== harmonized public ADME (modelling space) ===')
    for ep in endpoints:
        f = harmonize_endpoint(ep, config_path, v=v)
        f.to_parquet(f'{out_dir}/public_{ep}.parquet')
        res[ep] = f
    return res


if __name__ == '__main__':
    harmonize_all()
