"""Harvest the public `mdck_inh` dataset from ChEMBL: MDR1-MDCK Papp A->B WITH a P-gp inhibitor.

That is the passive-permeability leg of the in-house `mdck_inh` endpoint, and it is the only PUBLIC
source that measures the same quantity. Everything else (Biogen, TDC, Novartis) measures an efflux
ratio, a different cell line, or a predicted value — see wiki/wiki.md 2026-09-22.

Writes `<OUT_DIR>/<OUT_FILE>` with the four columns DATA._endpoint_dfs reads:
    compound  PUB_mdck_inh_<i>
    smiles    ChEMBL canonical SMILES
    value     log10(Papp / 1e-6 cm/s)  — MODELLING space, matching ADME_ENDPOINTS.mdck_inh
    origin    the config ORIGIN string

Run (env `ML`):
  python python/harvest_chembl_mdck_inh.py                 # use the cached API pages when present
  python python/harvest_chembl_mdck_inh.py --refresh       # re-query ChEMBL
  python python/harvest_chembl_mdck_inh.py --dry_run       # report the funnel, write nothing
Public ChEMBL data only. No in-house structure or value is sent anywhere.
"""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import yaml


class PARAMS():
    def __init__(self, config_path):
        self.config_path = config_path

    def load_params(self):
        """
        -Read the YAML config and expose every key as an attribute (e.g. params.ASSAY_RULES).
        return self:
        """
        with open(self.config_path) as f:
            self.__dict__.update(yaml.safe_load(f))
        print(f'> loaded {len(self.__dict__) - 1} params from {self.config_path}')
        return self


# ChEMBL resource -> the key holding its rows in the JSON response ('activity' is not a naive plural)
_ROWS_KEY = {'assay': 'assays', 'activity': 'activities', 'molecule': 'molecules'}


class DATA():
    def __init__(self):
        self.assays = None        # every ChEMBL assay whose description mentions the query term
        self.assay_ids = None     # the subset passing all ASSAY_RULES
        self.acts = None          # activities of those assays
        self.sel = None           # activities that are the target quantity
        self.smiles = None        # molecule_chembl_id -> canonical SMILES
        self.df = None            # the public-contract frame (compound/smiles/value/origin)
        self.funnel = {}          # step -> count, for the report

    def _get(self, params, resource, query, cache_name, refresh=False):
        """
        -Page one ChEMBL resource to exhaustion and cache the rows as JSON, so a re-run needs no network.
        param class params: PARAMS instance (API, CACHE_DIR, PAGE_SIZE)
        param str resource: 'assay' | 'activity' | 'molecule'
        param str query: the filter part of the query string
        param str cache_name: file name under params.CACHE_DIR
        param bool refresh: re-query instead of reading the cache
        return list: the resource's rows
        """
        os.makedirs(params.CACHE_DIR, exist_ok=True)
        path = os.path.join(params.CACHE_DIR, cache_name)
        if os.path.exists(path) and not refresh:
            with open(path) as f:
                rows = json.load(f)
            print(f'  cache hit {cache_name}: {len(rows)} rows')
            return rows
        # page until the offset passes the server's reported total
        rows, off = [], 0
        while True:
            url = f'{params.API}/{resource}.json?{query}&limit={params.PAGE_SIZE}&offset={off}'
            with urllib.request.urlopen(url, timeout=180) as r:
                d = json.load(r)
            rows += d[_ROWS_KEY[resource]]
            off += params.PAGE_SIZE
            if off >= d['page_meta']['total_count']:
                break
            time.sleep(0.2)
        with open(path, 'w') as f:
            json.dump(rows, f)
        print(f'  fetched {cache_name}: {len(rows)} rows')
        return rows

    def fetch_assays(self, params, refresh=False):
        """
        -Fetch every assay whose description contains params.ASSAY_QUERY (server-side filter).
        return None: (sets self.assays)
        """
        q = urllib.parse.urlencode({'description__icontains': params.ASSAY_QUERY})
        self.assays = [{'id': a['assay_chembl_id'], 'desc': (a['description'] or '').lower()}
                       for a in self._get(params, 'assay', q, 'assays.json', refresh)]
        self.funnel[f"description mentions '{params.ASSAY_QUERY}'"] = len(self.assays)

    def select_assays(self, params):
        """
        -Apply every ASSAY_RULES regex in order, then reject the reverse (B->A) direction. Each step's
         surviving count is recorded in self.funnel, so the report shows where the data is lost.
        return None: (sets self.assay_ids)
        """
        keep = self.assays
        for name, pat in params.ASSAY_RULES.items():
            rx = re.compile(pat)
            keep = [a for a in keep if rx.search(a['desc'])]
            self.funnel[f'  + {name}'] = len(keep)
        # an assay naming only the reverse leg is not the in-house quantity
        rx_bad = re.compile(params.EXCLUDE_DIRECTION)
        rx_ok = re.compile(params.ASSAY_RULES['direction'])
        keep = [a for a in keep if rx_ok.search(a['desc']) or not rx_bad.search(a['desc'])]
        self.funnel['  - reverse-direction assays'] = len(keep)
        self.assay_ids = [a['id'] for a in keep]
        self._desc = {a['id']: a['desc'] for a in keep}

    def fetch_activities(self, params, refresh=False):
        """
        -Fetch the activities of the selected assays, in batches of params.ID_BATCH assay ids.
        return None: (sets self.acts)
        """
        self.acts = []
        for i in range(0, len(self.assay_ids), params.ID_BATCH):
            batch = ','.join(self.assay_ids[i:i + params.ID_BATCH])
            q = urllib.parse.urlencode({'assay_chembl_id__in': batch})
            self.acts += self._get(params, 'activity', q, f'acts_{i:04d}.json', refresh)
        self.funnel['activities in those assays'] = len(self.acts)
        self.funnel['  distinct molecules'] = len({a['molecule_chembl_id'] for a in self.acts})

    def select_activities(self, params):
        """
        -Keep the activities that ARE the target quantity: a permeability value (not a ratio or a
         recovery), in a convertible unit, and not the reverse leg. Optionally drop censored rows.
        return None: (sets self.sel)
        """
        rx_type = re.compile(params.ACTIVITY_TYPE, re.I)
        rx_bad = re.compile(params.EXCLUDE_DIRECTION, re.I)
        # a permeability VALUE, by standard_type
        sel = [a for a in self.acts if a['standard_type'] and rx_type.match(a['standard_type'])]
        self.funnel['  permeability value (no ratio/recovery)'] = len(sel)
        # the reverse leg can also be named in the activity type itself
        sel = [a for a in sel if not rx_bad.search(a['standard_type'])]
        self.funnel['  - reverse-leg activity types'] = len(sel)
        # a unit the config can convert to 1e-6 cm/s
        sel = [a for a in sel if a['standard_units'] in params.UNIT_FACTOR and a['standard_value'] is not None]
        self.funnel['  + convertible unit'] = len(sel)
        # a censored value is a detection limit, not a measurement
        if params.DROP_CENSORED:
            sel = [a for a in sel if a['standard_relation'] == '=']
            self.funnel['  - censored (relation != "=")'] = len(sel)
        self.sel = sel

    def fetch_smiles(self, params, refresh=False):
        """
        -Fetch the canonical SMILES of every molecule left in self.sel.
        return None: (sets self.smiles)
        """
        mids = sorted({a['molecule_chembl_id'] for a in self.sel})
        rows = []
        for i in range(0, len(mids), params.ID_BATCH):
            q = urllib.parse.urlencode({'molecule_chembl_id__in': ','.join(mids[i:i + params.ID_BATCH])})
            rows += self._get(params, 'molecule', q, f'mols_{i:04d}.json', refresh)
        self.smiles = {m['molecule_chembl_id']: (m.get('molecule_structures') or {}).get('canonical_smiles')
                       for m in rows}
        self.funnel['  molecules with a SMILES'] = sum(v is not None for v in self.smiles.values())

    def build_frame(self, params):
        """
        -Convert every selected activity to the in-house modelling space and collapse duplicates.
         Papp is scaled to 1e-6 cm/s by the config UNIT_FACTOR, then transformed (log10). A molecule
         measured by several assays is aggregated by params.DEDUP.
        return None: (sets self.df — compound, smiles, value, origin)
        """
        # raw Papp in 1e-6 cm/s, per activity
        d = pd.DataFrame([{'mol': a['molecule_chembl_id'],
                           'papp': float(a['standard_value']) * params.UNIT_FACTOR[a['standard_units']]}
                          for a in self.sel])
        d['smiles'] = d['mol'].map(self.smiles)
        d = d.dropna(subset=['smiles'])
        # log10 needs a strictly positive Papp
        d = d[d['papp'] > 0]
        self.funnel['  + SMILES and Papp > 0'] = len(d)
        d['value'] = np.log10(d['papp']) if params.TRANSFORM == 'log10' else d['papp']
        # one row per molecule: several assays measure the same compound under different protocols
        g = d.groupby('smiles', as_index=False)['value'].agg(params.DEDUP)
        self.funnel[f'  unique molecules ({params.DEDUP} over assays)'] = len(g)
        g.insert(0, 'compound', [f'{params.ID_PREFIX}_{i}' for i in range(len(g))])
        g['origin'] = params.ORIGIN
        self.df = g[['compound', 'smiles', 'value', 'origin']]


class OUTPUT():
    def report(self, data, params):
        """
        -Print the selection funnel and the aggregate value distribution. Counts and percentiles only;
         no SMILES and no per-compound value is printed.
        return None:
        """
        print('\n> selection funnel')
        for k, v in data.funnel.items():
            print(f'  {k:46s} {v:7d}')
        v = data.df['value']
        print(f"\n> value ({params.TRANSFORM} of Papp / 1e-6 cm/s): n={len(v)} mean={v.mean():.3f} "
              f"std={v.std():.3f} min={v.min():.2f} p50={v.median():.2f} max={v.max():.2f}")

    def write_parquet(self, data, params):
        """
        -Write the public-contract parquet and assert the four columns the live path reads are present.
        return str: the output path
        """
        os.makedirs(params.OUT_DIR, exist_ok=True)
        path = os.path.join(params.OUT_DIR, params.OUT_FILE)
        # the contract DATA._endpoint_dfs reads — a missing column fails only at training time
        assert list(data.df.columns) == ['compound', 'smiles', 'value', 'origin'], data.df.columns.tolist()
        assert data.df['value'].notna().all() and np.isfinite(data.df['value']).all()
        assert data.df['compound'].is_unique and data.df['smiles'].is_unique
        data.df.to_parquet(path, index=False)
        print(f'\n> wrote {len(data.df)} rows -> {path}')
        return path


def MAIN():
    ap = argparse.ArgumentParser(description='Harvest the public mdck_inh dataset from ChEMBL.')
    ap.add_argument('--config', default='config/chembl_harvest.yaml', help='path to the YAML config')
    ap.add_argument('--refresh', action='store_true', help='re-query ChEMBL instead of reading the cache')
    ap.add_argument('--dry_run', action='store_true', help='report the funnel, write no parquet')
    args = ap.parse_args()

    params = PARAMS(args.config).load_params()
    data = DATA()
    data.fetch_assays(params, refresh=args.refresh)
    data.select_assays(params)
    data.fetch_activities(params, refresh=args.refresh)
    data.select_activities(params)
    data.fetch_smiles(params, refresh=args.refresh)
    data.build_frame(params)
    output = OUTPUT()
    output.report(data, params)
    if not args.dry_run:
        output.write_parquet(data, params)


if __name__ == '__main__':
    MAIN()
