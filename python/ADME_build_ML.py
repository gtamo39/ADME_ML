"""Build/evaluate the ADME single-task ML models (modularized from vignettes/Multitask_adme_preds.ipynb).

Mirror of python/MS_build_ML.py: a PARAMS/DATA/OUTPUT scaffold that runs standalone AND exposes every
step as a method callable from the notebook. The notebook keeps `params = PARAMS(cfg).load_params()`
then `data = DATA(); data.load_df_all(params)` — identical wiring to the __main__ block below.

Run (env `ML`):
  python python/ADME_build_ML.py                 # read the cached CDD pull
  python python/ADME_build_ML.py --overwrite     # re-pull df_all from CDD Vault and re-cache
Aggregate output only — no SMILES / compound IDs / per-compound values are printed.
"""
import os, sys
# self-locate repo root (parent of this file's dir) so `import python.functions` and relative
# paths (config/, data/, output/) resolve no matter where the script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_REPO_ROOT), 'Scripts'))  # shared helpers (Rdkit_tools, ML_Reg, ...)
sys.path.insert(0, os.path.expanduser('~/Scripts'))                       # shared helpers (home checkout)
sys.path.insert(0, os.path.expanduser('~/CDD_Vault_API/python'))          # CDD Vault API (get_protocol_data)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from glob import glob
from datetime import date
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

# local modules
import python.functions as fn
import Rdkit_tools as rdkit_tools
import ML_Reg as ML_Reg
import ML_Class as ML_Class
import Statistics_tools as stats_tools
from get_protocol_data import get_data, load_config, alias_map   # CDD Vault protocol export

# raw <-> modelling-space transforms per endpoint (config ADME_ENDPOINTS[<ep>]['transform']): (forward, inverse)
_TF = {
    'log10':     (lambda v: np.log10(v),               lambda l: 10.0 ** l),
    'identity':  (lambda v: v,                          lambda l: l),
    'logit_pct': (lambda v: np.log10(v / (100.0 - v)),  lambda l: 100.0 / (1.0 + 10.0 ** (-l))),
}


class PARAMS():
    def __init__(self, config_path):
        self.config_path = config_path

    def load_params(self):
        """
        -Read the YAML config and expose every key as an attribute (e.g. params.ADME_ENDPOINTS).
        -Bind the CDD Vault paths (config override wins over the ~/CDD_Vault_API defaults).
        return self:
        """
        with open(self.config_path) as f:
            self.__dict__.update(yaml.safe_load(f))
        self.CDD_CONFIG = getattr(self, 'CDD_CONFIG', os.path.expanduser('~/CDD_Vault_API/config/config.yaml'))
        self.CDD_TOKEN  = getattr(self, 'CDD_TOKEN',  os.path.expanduser('~/.cdd_token'))
        self.ADME_RAW_CSV = getattr(self, 'ADME_RAW_CSV', 'data/20260707_all_adme.csv')
        self.ADME_CACHE = getattr(self, 'ADME_CACHE', 'autoresearch/predict_adme')   # sanitized public parquets
        print(f'> loaded {len(self.__dict__) - 1} params from {self.config_path}')
        return self


class DATA():
    def __init__(self):
        self.df_internal_exp_all = None   # raw internal experimental CDD pull (one row per compound, every endpoint column)
        self.dfs = {}               # endpoint -> harmonized augmented dataset (internal + public)
        self.MF_features = {}       # endpoint -> H237 feature matrix (compound + features)
        self.ML_data = {}           # endpoint -> modelling frame (features + label + meta)

    def load_df_internal_exp_all(self, params, overwrite=False):
        """
        -Load the internal experimental ADME protocol data (name + smiles + every endpoint column). With
         overwrite, pull the latest from CDD Vault and cache to params.ADME_RAW_CSV; otherwise read the cache.
        param class params: PARAMS instance (CDD_CONFIG, CDD_TOKEN, ADME_RAW_CSV)
        param bool overwrite: re-pull from CDD Vault instead of reading the cached csv
        return None:
        """
        if overwrite:
            alias_map(load_config(params.CDD_CONFIG))                       # register the human-readable column aliases
            self.df_internal_exp_all = get_data(config_path=params.CDD_CONFIG, token_file=params.CDD_TOKEN)
            self.df_internal_exp_all.to_csv(params.ADME_RAW_CSV, index=False)
        else:
            self.df_internal_exp_all = pd.read_csv(params.ADME_RAW_CSV)
        print(f'> df_internal_exp_all: {self.df_internal_exp_all.shape[0]} compounds x {self.df_internal_exp_all.shape[1]} columns')

    def load_combine_dfs(self, params):
        """
        -Build every endpoint's augmented dataset (self.dfs[ep], via each get_<ep>_data) and combine them into
         one WIDE frame self.df_all: one row per compound, one column per endpoint holding its label (modelling
         space; NaN where the compound has no measurement for that endpoint), plus smiles/source/origin. Public
         compound ids are consistent across endpoints, so a molecule measured for several endpoints is one row.
        param class params: PARAMS instance (ADME_ENDPOINTS, ADME_CACHE, ENDPOINT_PUBLIC_FILES)
        return None: (self.df_all — compound, smiles, <endpoint labels...>, source, origin)
        """
        eps = list(params.ADME_ENDPOINTS)
        # build any endpoint dfs not already present
        for ep in eps:
            if ep not in self.dfs:
                getattr(self, f'get_{ep}_data')(params)
        # long -> wide: one label column per endpoint, collapsed per compound (first non-null)
        frames = [self.dfs[ep][['compound', 'smiles', 'source', 'origin', 'label']].rename(columns={'label': ep}) for ep in eps]
        df = pd.concat(frames, ignore_index=True).groupby('compound', as_index=False).first()
        self.df_all = df[['compound', 'smiles'] + eps + ['source', 'origin']]
        n_meas = int(self.df_all[eps].notna().any(axis=1).sum())
        print(f"> df_all: {self.df_all.shape[0]} compounds x {self.df_all.shape[1]} cols ({len(eps)} endpoints; {n_meas} with >=1 label)")

    def build_MF_features(self, params, type='H237', date='20260824', n_jobs=32):
        """
        -Build (or load if present) the molecular-feature matrix for EVERY compound in self.df_all (all
         endpoints, internal + public). 'H237' = H236 fingerprints/physchem/MACCS/AtomPair + ~200 descriptastorus
         DS_ descriptors; 'H236' = the fingerprint block only. Cached to
         params.MF_features_all_path/<date>_MF_features.parquet; stored in self.MF_features['all'].
        param class params: PARAMS instance (MF_features_all_path)
        param str type: 'H237' | 'H236'
        param str date: filename stamp for the cache parquet
        param int n_jobs: processes for the H237 descriptor block
        return None: (self.MF_features['all'] — compound + feature columns)
        """
        os.makedirs(params.MF_features_all_path, exist_ok=True)
        path = os.path.join(params.MF_features_all_path, f'{date}_MF_features.parquet')
        if os.path.exists(path):
            self.MF_features['all'] = pd.read_parquet(path)
        else:
            smi = self.df_all[['compound', 'smiles']]
            if type == 'H237':
                self.MF_features['all'] = rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
            elif type == 'H236':
                self.MF_features['all'] = rdkit_tools.compute_H236_features(smi, v=True)
            else:
                raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")
            self.MF_features['all'].to_parquet(path)
        print(f"> MF features ({type}) [all]: {self.MF_features['all'].shape[0]} compounds x "
              f"{self.MF_features['all'].shape[1] - 1} features -> {path}")

    def _feats(self, endpoint):
        """Return the endpoint's own feature matrix if built, else the unified MF_features['all'] (from build_MF_features)."""
        return self.MF_features[endpoint] if endpoint in self.MF_features else self.MF_features['all']

    def get_solubility_data(self, params):
        """
        -Build the solubility augmented dataset: internal thermodynamic solubility (from df_internal_exp_all,
         raw µM -> log10 via the config transform) + the final sanitized public EXP set (augmented_sources
         solubility == ['EXP']). Add a raw-µM column. Store in self.dfs['solubility'].
        param class params: PARAMS instance (ADME_ENDPOINTS, ADME_CACHE)
        return None: (result stored in self.dfs['solubility'] — compound, smiles, label[log10 µM], source, origin, raw[µM])
        """
        sc = params.ADME_ENDPOINTS['solubility']['col']                    # CDD µM readout (plain col is µg/mL)
        internal = (self.df_internal_exp_all[['name', 'smiles', sc]].dropna(subset=[sc])
                    .rename(columns={'name': 'compound', sc: 'label'}))
        internal['label'] = np.log10(internal['label'].astype(float))
        internal['source'] = 'internal'; internal['origin'] = 'internal'
        exp = (pd.read_parquet(f'{params.ADME_CACHE}/public_solubility.parquet',
                               columns=['compound', 'smiles', 'value', 'origin']).rename(columns={'value': 'label'}))
        exp['source'] = 'EXP'
        df = pd.concat([internal, exp[internal.columns]], ignore_index=True)
        df['raw'] = 10.0 ** df['label']                                    # raw µM (inverse of the log10 label)
        self.dfs['solubility'] = df
        print(f"> solubility augmented: {len(df)} rows (internal={len(internal)}, EXP={len(exp)})")

    def build_MF_features_solubility(self, type='H237', path=None, n_jobs=32):
        """
        -Build (or load from `path`) the molecular-feature matrix for the solubility compounds. 'H237' =
         H236 (Morgan + physchem + MACCS + AtomPair) + ~200 descriptastorus DS_ descriptors; 'H236' = the
         fingerprint block only. Computed once and cached to `path`; stored in self.MF_features['solubility'].
        param str type: 'H237' | 'H236'
        param str path: parquet cache path (load if it exists, else compute + save there)
        param int n_jobs: processes for the H237 descriptor block (~32 is optimal, see compute_H237_features)
        return None: (result stored in self.MF_features['solubility'] — compound + feature columns)
        """
        if path and os.path.exists(path):
            self.MF_features['solubility'] = pd.read_parquet(path)
        else:
            smi = self.dfs['solubility'][['compound', 'smiles']]
            if type == 'H237':
                self.MF_features['solubility'] = rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
            elif type == 'H236':
                self.MF_features['solubility'] = rdkit_tools.compute_H236_features(smi, v=True)
            else:
                raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.MF_features['solubility'].to_parquet(path)

        print(f"> MF features ({type}): {self.MF_features['solubility'].shape[0]} compounds x {self.MF_features['solubility'].shape[1] - 1} features")

    def build_ML_data_solubility(self, min_raw_uM=0.0, sol_cap_uM=15000.0):
        """
        -Assemble the solubility modelling frame: merge the H237 features onto the augmented labels, clip the
         label at sol_cap_uM (µM; winsorize saturated high solubility), keep only physical concentrations
         (raw µM >= min_raw_uM), drop feature NaNs. Store in self.ML_data['solubility'].
        param float min_raw_uM: keep rows whose raw solubility >= this (µM); 0.0 keeps every positive value
        param float sol_cap_uM: upper clip on the label, in µM (saturated high solubility carries little signal)
        return None: (result stored in self.ML_data['solubility'] — compound, smiles, label, source, origin + H237 features)
        """
        d = self.dfs['solubility'][['compound', 'smiles', 'label', 'source', 'origin']].copy()
        d['label'] = d['label'].clip(upper=np.log10(sol_cap_uM))            # cap saturated high solubility
        d = pd.merge(self._feats('solubility'), d, on='compound').dropna()
        # d = d[(10.0 ** d['label']) >= min_raw_uM]                           # keep physical concentrations (raw µM >= min_raw_uM)
        d = (d.assign(_int=(d['source'] == 'internal'))                     # dedup molecules: prefer the internal row on a SMILES clash
               .sort_values('_int', ascending=False, kind='stable')
               .drop_duplicates('smiles', keep='first').drop(columns='_int'))
        meta = ['compound', 'smiles', 'label', 'source', 'origin']
        # const = [c for c in d.columns if c not in meta and d[c].nunique() <= 1]
        # d = d.drop(columns=const)                                          # drop zero-variance (constant) feature columns
        self.ML_data['solubility'] = d.reset_index(drop=True)
        print(f"> ML_data['solubility']: {self.ML_data['solubility'].shape[0]} rows x {self.ML_data['solubility'].shape[1]} cols "
              f"(cap {sol_cap_uM:g} µM, min {min_raw_uM:g} µM")

    def get_logd_data(self, params):
        """
        -Build the logD augmented dataset: internal LogD7.4 (from df_internal_exp_all) + the sanitized public
         EXP set (public_logd.parquet). LogD is already log-scale (config transform 'identity'), so the label
         is the raw value and `raw` == `label`. Store in self.dfs['logd'].
        param class params: PARAMS instance (ADME_ENDPOINTS, ADME_CACHE)
        return None: (result stored in self.dfs['logd'] — compound, smiles, label[logD7.4], source, origin, raw[logD7.4])
        """
        lc = params.ADME_ENDPOINTS['logd']['col']                          # CDD LogD7.4 readout (identity transform)
        internal = (self.df_internal_exp_all[['name', 'smiles', lc]].dropna(subset=[lc])
                    .rename(columns={'name': 'compound', lc: 'label'}))
        internal['label'] = internal['label'].astype(float)                # identity: no log transform
        internal['source'] = 'internal'; internal['origin'] = 'internal'
        exp = (pd.read_parquet(f'{params.ADME_CACHE}/public_logd.parquet',
                               columns=['compound', 'smiles', 'value', 'origin']).rename(columns={'value': 'label'}))
        exp['source'] = 'EXP'
        df = pd.concat([internal, exp[internal.columns]], ignore_index=True)
        df['raw'] = df['label']                                            # identity transform: raw == label
        self.dfs['logd'] = df
        print(f"> logd augmented: {len(df)} rows (internal={len(internal)}, EXP={len(exp)})")

    def build_MF_features_logd(self, type='H237', path=None, n_jobs=32):
        """
        -Build (or load from `path`) the molecular-feature matrix for the logD compounds. Same feature block as
         solubility ('H237' = H236 + ~200 descriptastorus DS_). Computed once and cached; stored in
         self.MF_features['logd'].
        param str type: 'H237' | 'H236'
        param str path: parquet cache path (load if it exists, else compute + save there)
        param int n_jobs: processes for the H237 descriptor block
        return None: (result stored in self.MF_features['logd'] — compound + feature columns)
        """
        if path and os.path.exists(path):
            self.MF_features['logd'] = pd.read_parquet(path)
        else:
            smi = self.dfs['logd'][['compound', 'smiles']]
            if type == 'H237':
                self.MF_features['logd'] = rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
            elif type == 'H236':
                self.MF_features['logd'] = rdkit_tools.compute_H236_features(smi, v=True)
            else:
                raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.MF_features['logd'].to_parquet(path)
        print(f"> MF features ({type}): {self.MF_features['logd'].shape[0]} compounds x {self.MF_features['logd'].shape[1] - 1} features")

    def build_ML_data_logd(self):
        """
        -Assemble the logD modelling frame: merge the H237 features onto the augmented labels, drop feature NaNs,
         and dedup molecules by SMILES (prefer the internal row on a clash). LogD needs no floor/cap (identity
         scale, physical range). Store in self.ML_data['logd'].
        return None: (result stored in self.ML_data['logd'] — compound, smiles, label, source, origin + H237 features)
        """
        d = self.dfs['logd'][['compound', 'smiles', 'label', 'source', 'origin']].copy()
        d = pd.merge(self._feats('logd'), d, on='compound').dropna()
        d = (d.assign(_int=(d['source'] == 'internal'))                    # dedup molecules: prefer the internal row on a SMILES clash
               .sort_values('_int', ascending=False, kind='stable')
               .drop_duplicates('smiles', keep='first').drop(columns='_int'))
        self.ML_data['logd'] = d.reset_index(drop=True)
        print(f"> ML_data['logd']: {self.ML_data['logd'].shape[0]} rows x {self.ML_data['logd'].shape[1]} cols")

    def get_mdck_data(self, params):
        """
        -Build the MDCK permeability augmented dataset: internal MDR1 Papp A->B (from df_internal_exp_all,
         raw 1e-6 cm/s -> log10 via the config transform, keeping only the MDR1 cell line per the config filter)
         + the sanitized public EXP sets (Novartis-NIBR + ADMETlab-PROTAC, value already log10 Papp). Add a raw
         column. Store in self.dfs['mdck'].
        param class params: PARAMS instance (ADME_ENDPOINTS, ADME_CACHE)
        return None: (result stored in self.dfs['mdck'] — compound, smiles, label[log10 Papp], source, origin, raw[Papp])
        """
        ep = params.ADME_ENDPOINTS['mdck']; mc = ep['col']                 # CDD Papp A->B readout (1e-6 cm/s)
        df0 = self.df_internal_exp_all
        filt = ep.get('filter')                                            # keep only the MDR1 cell line
        if filt:
            df0 = df0[df0[filt['col']].astype(str).str.contains(filt['contains'], na=False)]
        internal = (df0[['name', 'smiles', mc]].dropna(subset=[mc])
                    .rename(columns={'name': 'compound', mc: 'label'}))
        internal['label'] = np.log10(internal['label'].astype(float))
        internal['source'] = 'internal'; internal['origin'] = 'internal'
        exp = pd.concat([pd.read_parquet(f'{params.ADME_CACHE}/{f}', columns=['compound', 'smiles', 'value', 'origin'])
                         for f in ['public_novartis_mdck.parquet', 'public_admetlab_mdck.parquet']],
                        ignore_index=True).rename(columns={'value': 'label'})   # value already log10 Papp
        exp['source'] = 'EXP'
        df = pd.concat([internal, exp[internal.columns]], ignore_index=True)
        df['raw'] = 10.0 ** df['label']                                    # raw Papp (inverse of the log10 label)
        self.dfs['mdck'] = df
        print(f"> mdck augmented: {len(df)} rows (internal={len(internal)}, EXP={len(exp)})")

    def build_MF_features_mdck(self, type='H237', path=None, n_jobs=32):
        """
        -Build (or load from `path`) the molecular-feature matrix for the MDCK compounds. Same feature block as
         solubility/logD ('H237' = H236 + ~200 descriptastorus DS_). Computed once and cached; stored in
         self.MF_features['mdck'].
        param str type: 'H237' | 'H236'
        param str path: parquet cache path (load if it exists, else compute + save there)
        param int n_jobs: processes for the H237 descriptor block
        return None: (result stored in self.MF_features['mdck'] — compound + feature columns)
        """
        if path and os.path.exists(path):
            self.MF_features['mdck'] = pd.read_parquet(path)
        else:
            smi = self.dfs['mdck'][['compound', 'smiles']]
            if type == 'H237':
                self.MF_features['mdck'] = rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
            elif type == 'H236':
                self.MF_features['mdck'] = rdkit_tools.compute_H236_features(smi, v=True)
            else:
                raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.MF_features['mdck'].to_parquet(path)
        print(f"> MF features ({type}): {self.MF_features['mdck'].shape[0]} compounds x {self.MF_features['mdck'].shape[1] - 1} features")

    def build_ML_data_mdck(self):
        """
        -Assemble the MDCK modelling frame: merge the H237 features onto the augmented labels, drop feature NaNs,
         and dedup molecules by SMILES (prefer the internal row on a clash). No floor/cap. Store in self.ML_data['mdck'].
        return None: (result stored in self.ML_data['mdck'] — compound, smiles, label, source, origin + H237 features)
        """
        d = self.dfs['mdck'][['compound', 'smiles', 'label', 'source', 'origin']].copy()
        d = pd.merge(self._feats('mdck'), d, on='compound').dropna()
        d = d[np.isfinite(d['label'])]                                     # drop non-finite labels (log10 of a 0/negative Papp)
        d = (d.assign(_int=(d['source'] == 'internal'))                    # dedup molecules: prefer the internal row on a SMILES clash
               .sort_values('_int', ascending=False, kind='stable')
               .drop_duplicates('smiles', keep='first').drop(columns='_int'))
        self.ML_data['mdck'] = d.reset_index(drop=True)
        print(f"> ML_data['mdck']: {self.ML_data['mdck'].shape[0]} rows x {self.ML_data['mdck'].shape[1]} cols")

    # ---- generic, config-driven builders (used by hlm/mlm/rlm/caco2/ppb via the aliases below) ----
    def _endpoint_dfs(self, params, endpoint):
        """
        -Generic augmented-dataset builder: internal endpoint column mapped to modelling space via the config
         transform (log10 / identity / logit_pct), plus the concatenated public EXP parquets listed in
         params.ENDPOINT_PUBLIC_FILES[endpoint] (their 'value' is already in modelling space). Store self.dfs[endpoint].
        return None: (self.dfs[endpoint] — compound, smiles, label, source, origin, raw)
        """
        ep = params.ADME_ENDPOINTS[endpoint]; col = ep['col']; fwd, inv = _TF[ep['transform']]
        df0 = self.df_internal_exp_all
        filt = ep.get('filter')                                            # optional cell-line / assay filter (e.g. mdck MDR1)
        if filt:
            df0 = df0[df0[filt['col']].astype(str).str.contains(filt['contains'], na=False)]
        internal = (df0[['name', 'smiles', col]].dropna(subset=[col]).rename(columns={'name': 'compound', col: 'label'}))
        internal['label'] = fwd(internal['label'].astype(float))           # raw -> modelling space
        internal['source'] = 'internal'; internal['origin'] = 'internal'
        exp = pd.concat([pd.read_parquet(f"{params.ADME_CACHE}/{f}", columns=['compound', 'smiles', 'value', 'origin'])
                         for f in params.ENDPOINT_PUBLIC_FILES[endpoint]], ignore_index=True).rename(columns={'value': 'label'})
        exp['source'] = 'EXP'
        df = pd.concat([internal, exp[internal.columns]], ignore_index=True)
        df['raw'] = inv(df['label'])                                       # inverse transform -> raw units
        self.dfs[endpoint] = df
        print(f"> {endpoint} augmented: {len(df)} rows (internal={len(internal)}, EXP={len(exp)})")

    def _endpoint_MF(self, endpoint, type='H237', path=None, n_jobs=32):
        """Generic H237/H236 feature builder (load `path` if it exists, else compute + cache). -> self.MF_features[endpoint]."""
        if path and os.path.exists(path):
            self.MF_features[endpoint] = pd.read_parquet(path)
        else:
            smi = self.dfs[endpoint][['compound', 'smiles']]
            if type == 'H237':
                self.MF_features[endpoint] = rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
            elif type == 'H236':
                self.MF_features[endpoint] = rdkit_tools.compute_H236_features(smi, v=True)
            else:
                raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.MF_features[endpoint].to_parquet(path)
        print(f"> MF features ({type}): {self.MF_features[endpoint].shape[0]} compounds x {self.MF_features[endpoint].shape[1] - 1} features")

    def _endpoint_ML(self, endpoint):
        """Generic modelling frame: merge features + labels, drop NaN + non-finite labels, dedup by SMILES (prefer internal)."""
        d = self.dfs[endpoint][['compound', 'smiles', 'label', 'source', 'origin']].copy()
        d = pd.merge(self._feats(endpoint), d, on='compound').dropna()
        d = d[np.isfinite(d['label'])]                                     # drop non-finite labels (log10/logit of a 0/edge value)
        d = (d.assign(_int=(d['source'] == 'internal'))                    # dedup molecules: prefer the internal row on a SMILES clash
               .sort_values('_int', ascending=False, kind='stable')
               .drop_duplicates('smiles', keep='first').drop(columns='_int'))
        self.ML_data[endpoint] = d.reset_index(drop=True)
        print(f"> ML_data['{endpoint}']: {self.ML_data[endpoint].shape[0]} rows x {self.ML_data[endpoint].shape[1]} cols")


# generate config-driven per-endpoint aliases so the notebook's exec('data.get_'+k+'_data(...)') pattern works
# (hlm/mlm/rlm/caco2/ppb route through the generic _endpoint_* builders; solubility/logd/mdck keep bespoke methods)
for _ep in ('hlm', 'mlm', 'rlm', 'caco2', 'ppb'):
    setattr(DATA, f'get_{_ep}_data',          lambda self, params, _e=_ep: self._endpoint_dfs(params, _e))
    setattr(DATA, f'build_MF_features_{_ep}', lambda self, type='H237', path=None, n_jobs=32, _e=_ep: self._endpoint_MF(_e, type, path, n_jobs))
    setattr(DATA, f'build_ML_data_{_ep}',     lambda self, _e=_ep: self._endpoint_ML(_e))


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build/evaluate the ADME single-task ML models.")
    ap.add_argument('--config', default='config/config.yaml', help="path to the YAML config")
    ap.add_argument('--overwrite', action='store_true', help="re-pull df_all from CDD Vault and re-cache")
    args = ap.parse_args()

    ## params:
    params = PARAMS(args.config)
    params.load_params()

    ## data:
    data = DATA()
    data.load_df_internal_exp_all(params, overwrite=args.overwrite)
