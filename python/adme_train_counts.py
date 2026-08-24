"""Training-row counts per prediction arm for the Summary_results notebook (aggregate only; NO model fits).

For every endpoint it recomputes, from the same loaders the sweep/temporal runs use:
  - n_internal  : internal compounds measured for the endpoint
  - n_public    : cleaned public rows the arm adds to training (CV: per cleaning-sweep strategy; temporal: the
                  pooled public, capped at 40000 exactly as temporal_eval does)
  - n_train     : total training pool = n_internal + n_public   (CV arms train on all internal; temporal arms
                  train on the oldest 90% window union -> int(0.9*n_internal) + n_public)
  - providers   : which data providers the public sources come from (biogen / AstraZeneca / Novartis / ADMETlab)

Keys match the pred_<ep> dict keys in vignettes/Summary_results.ipynb (read per-endpoint _LBL/_TMAP from the
notebook so the labels line up). Output: output/train_counts.csv. Reads SMILES-bearing files locally but only
ever writes counts. Run (env ML):  python python/adme_train_counts.py
"""
import ast, json, os, re, sys
from pathlib import Path
import pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'python')); sys.path.insert(0, os.path.expanduser('~/Scripts'))
import adme_cleaning_sweep as SW
from adme_cleaning_sweep import load, clean_public, STRATEGIES, strat_signature, CACHE, PUBLIC_SRC, _feats
from run_RF_SingleTask_systematic import PARAMS, DATA
from temporal_eval import arms_for, ANCHORS

NB = ROOT / 'vignettes/Summary_results.ipynb'
TEMP_CAP = 40000                                   # temporal_eval caps TOTAL pooled public at this
ANCHOR_MAX = ANCHORS[-1]                            # rolling train union = oldest 90% internal
PROV = {'NVS': 'Novartis', 'ADM': 'ADMETlab'}      # predicted sources
EXP_PROV = {                                        # experimental providers per endpoint (aggregate origins)
    'solubility': 'PharmaBench/ChEMBL/Biogen(+)', 'logd': 'AstraZeneca', 'hlm': 'Biogen/AstraZeneca',
    'rlm': 'Biogen', 'caco2': 'Wang(TDC)', 'ppb': 'AstraZeneca/Biogen'}


def providers(ep, srcs):
    """Human-readable provider string for a source-type set (order EXP, NVS, ADM)."""
    parts = ([EXP_PROV.get(ep, 'experimental')] if 'EXP' in srcs else []) + [PROV[s] for s in ('NVS', 'ADM') if s in srcs]
    return ' + '.join(parts) if parts else 'internal only'


def nb_maps(ep):
    """(_LBL, _TMAP) dicts from the endpoint's load cell so keys match the notebook exactly."""
    nb = json.load(open(NB)); tag = ep if ep != 'caco2' else 'Caco-2'
    for c in nb['cells']:
        s = ''.join(c['source'])
        if s.startswith('## load prediction dfs') and (f'pred_{ep} =' in s):
            lbl = ast.literal_eval(re.search(r'_LBL\s*=\s*(\{.*?\})', s).group(1))
            tm = ast.literal_eval(re.search(r'_TMAP\s*=\s*(\{.*?\})', s).group(1))
            return lbl, tm
    return {}, {}


def sol_rows(data, params_obj):
    """Solubility has a bespoke load cell (solsweep strategies, thermo-promoted). CV n_public from the
    stable solsweep log; temporal via data.pooled (thermo public, capped)."""
    n_int = len(data.internal_ep('solubility')); base_t = int(n_int * ANCHOR_MAX)
    n_pub_t = min(sum(len(p) for p in data.pooled('solubility', data.augmented_sources('solubility', params_obj))[2]), TEMP_CAP)
    ep = 'solubility'
    return [
        {'endpoint': ep, 'key': 'internal_cv', 'sources': 'none', 'providers': 'internal only', 'n_internal': n_int, 'n_public': 0, 'n_train': n_int},
        {'endpoint': ep, 'key': 'EXP_thermo_cv', 'sources': 'EXP', 'providers': EXP_PROV[ep], 'n_internal': n_int, 'n_public': 17869, 'n_train': n_int + 17869},
        {'endpoint': ep, 'key': 'contaminated_cv', 'sources': 'EXP(all)', 'providers': 'mixed (pre-cleanup)', 'n_internal': n_int, 'n_public': 106307, 'n_train': n_int + 106307},
        {'endpoint': ep, 'key': 'EXP_thermo_temporal', 'sources': 'EXP', 'providers': EXP_PROV[ep], 'n_internal': base_t, 'n_public': n_pub_t, 'n_train': base_t + n_pub_t},
        {'endpoint': ep, 'key': 'internal_only_temporal', 'sources': 'none', 'providers': 'internal only', 'n_internal': base_t, 'n_public': 0, 'n_train': base_t},
    ]


def endpoint_rows(ep, mf, feats, params, data, params_obj):
    knobs = {'lo': params['phys'][ep][0], 'hi': params['phys'][ep][1],
             'wlo': params['winsor'][0], 'whi': params['winsor'][1], 'k': params['iqr_k']}
    internal, pub, iks = load(ep, mf, feats, params['cap'], params['seed'])
    n_int = len(internal); lbl, tm = nb_maps(ep); rows = []

    # internal-only CV (loaded from predictions_runs) — no public
    rows.append({'endpoint': ep, 'key': 'internal_cv', 'sources': 'none', 'providers': 'internal only',
                 'n_internal': n_int, 'n_public': 0, 'n_train': n_int})
    # CV cleaning-sweep strategies (dedup degenerate ones exactly as the sweep does)
    seen = {}
    for strat in STRATEGIES:
        sig = strat_signature(pub, strat, knobs, iks)
        if sig in seen:
            continue
        seen[sig] = strat
        n_pub = len(clean_public(pub, strat, knobs, iks)) if len(pub) else 0
        srcs = set(sig[0])
        rows.append({'endpoint': ep, 'key': 'cv_' + lbl.get(strat, strat), 'sources': '+'.join(sorted(srcs)) or 'none',
                     'providers': providers(ep, srcs), 'n_internal': n_int, 'n_public': n_pub, 'n_train': n_int + n_pub})

    # temporal rolling-origin arms (train union = oldest 90% internal + pooled public capped at 40000)
    n_int_t = len(data.internal_ep(ep)); base_t = int(n_int_t * ANCHOR_MAX)
    for name, srcs in arms_for(ep, data, params_obj):
        _, _, pub_list = data.pooled(ep, srcs)
        n_pub = min(sum(len(p) for p in pub_list), TEMP_CAP)
        rows.append({'endpoint': ep, 'key': 'temporal_' + tm.get(name, name), 'sources': '+'.join(srcs) or 'none',
                     'providers': providers(ep, set(srcs)), 'n_internal': base_t, 'n_public': n_pub, 'n_train': base_t + n_pub})
    return rows


if __name__ == '__main__':
    cfg = yaml.safe_load((ROOT / 'config/config.yaml').read_text())
    rf, sw = cfg['RF_SINGLETASK'], cfg['ADME_CLEAN_SWEEP']
    params = {'seed': rf['seed'], 'cap': sw['predicted_cap'], 'winsor': sw['winsor_pct'],
              'iqr_k': sw['iqr_k'], 'phys': sw['phys_range']}
    params_obj = PARAMS(); data = DATA().load_all(params_obj)
    mf, feats = _feats()
    endpoints = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']
    out = []
    for ep in endpoints:
        try:
            out += sol_rows(data, params_obj) if ep == 'solubility' else endpoint_rows(ep, mf, feats, params, data, params_obj)
            print(f'{ep:11} ok', flush=True)
        except Exception as e:
            print(f'{ep:11} FAIL {type(e).__name__}: {e}', flush=True)
    df = pd.DataFrame(out)
    (ROOT / 'output').mkdir(exist_ok=True)
    df.to_csv(ROOT / 'output/train_counts.csv', index=False)
    print(f'\n> wrote output/train_counts.csv  ({len(df)} rows)', flush=True)
    print(df.to_string(index=False), flush=True)
