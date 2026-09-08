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
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from glob import glob
from datetime import date
from itertools import combinations
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
        self.params = None          # PARAMS instance (stored on load; read by the generic builders)
        self.k = None               # current endpoint key (get_internal_public_sets)
        self.d = self.internal = self.pub = None   # last endpoint's split sets (get_internal_public_sets)
        self.origins = None         # public origins with > min_n compounds (get_internal_public_sets)
        self.combo = None           # chosen public origins for the deployed model (select_best_combo_and_update)
        self.pub_ids = self.int_ids = None            # compound ids (select_best_combo_and_update)
        self.fold_ids = self.fold_ids_aug = None      # CV splits, internal + augmented (select_best_combo_and_update)
        self.train_temp_ids = self.test_temp_ids = None   # 80/20 temporal split (select_best_combo_and_update)
        self.cp_k = self.cp_grouping = self.cp_tasks = self.cp_ds_cols = None   # chemprop arm (build_ML_data_CP)
        self.cp_pub = self.cp_int = None              # chemprop pretrain / finetune pools (build_ML_data_CP)
        self.cp_folds = self.cp_fold_source = None    # chemprop CV folds + where they came from (build_ML_data_CP)

    def load_df_internal_exp_all(self, params, overwrite=False):
        """
        -Load the internal experimental ADME protocol data (name + smiles + every endpoint column). With
         overwrite, pull the latest from CDD Vault and cache to params.ADME_RAW_CSV; otherwise read the cache.
        param class params: PARAMS instance (CDD_CONFIG, CDD_TOKEN, ADME_RAW_CSV)
        param bool overwrite: re-pull from CDD Vault instead of reading the cached csv
        return None:
        """
        self.params = params                                                # keep for the generic builders (label caps, etc.)
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
        -Build the molecular-feature matrix for EVERY compound in self.df_all, in two parts.
         PUBLIC compounds are CACHED at params.MF_features_all_path/<date>_MF_features_public.parquet, because
         their SMILES never change; the cache is topped up for any public compound it does not carry yet.
         INTERNAL compounds are ALWAYS RECOMPUTED, because every CDD pull adds new ones and a cached matrix
         would silently miss them (the inner merge in _endpoint_ML then drops those rows without a word).
         The two blocks concatenate into self.MF_features['all'], and a coverage assertion proves that every
         df_all compound has features. 'H237' = H236 fingerprints/physchem/MACCS/AtomPair + ~200
         descriptastorus DS_ descriptors; 'H236' = the fingerprint block only.
        param class params: PARAMS instance (MF_features_all_path)
        param str type: 'H237' | 'H236'
        param str date: filename stamp for the public cache parquet
        param int n_jobs: processes for the H237 descriptor block
        return None: (self.MF_features['all'] — compound + feature columns)
        """
        os.makedirs(params.MF_features_all_path, exist_ok=True)
        pub_path = os.path.join(params.MF_features_all_path, f'{date}_MF_features_{type}_public.parquet')
        is_int = self.df_all['source'] == 'internal'
        pub_ids = set(self.df_all.loc[~is_int, 'compound'])
        # public block: read the cache, or derive it once from the legacy all-compound parquet, or compute it
        legacy = os.path.join(params.MF_features_all_path, f'{date}_MF_features.parquet')
        if os.path.exists(pub_path):
            pub = pd.read_parquet(pub_path)
        elif os.path.exists(legacy):
            pub = pd.read_parquet(legacy)
            pub = pub[pub['compound'].isin(pub_ids)].reset_index(drop=True)
            pub.to_parquet(pub_path)
            print(f'> MF public cache derived from {legacy}: {len(pub)} public compounds -> {pub_path}')
        else:
            pub = self._compute_MF(self.df_all.loc[~is_int, ['compound', 'smiles']], type, n_jobs)
            pub.to_parquet(pub_path)
        # top up the public cache for public compounds it does not hold yet (a new public file was added)
        new_pub = self.df_all.loc[~is_int & ~self.df_all['compound'].isin(set(pub['compound'])),
                                  ['compound', 'smiles']]
        if len(new_pub):
            pub = pd.concat([pub, self._compute_MF(new_pub, type, n_jobs)], ignore_index=True)
            pub.to_parquet(pub_path)
            print(f'> MF public cache topped up with {len(new_pub)} new public compounds')
        # internal block: ALWAYS recomputed, so the newest CDD pull is always covered
        internal = self._compute_MF(self.df_all.loc[is_int, ['compound', 'smiles']], type, n_jobs)
        assert set(internal.columns) == set(pub.columns), (
            f'internal and public feature columns differ — the public cache holds another feature version; '
            f'delete {pub_path} and rebuild')
        self.MF_features['all'] = pd.concat([pub, internal[pub.columns]], ignore_index=True)
        # coverage: a missing compound would be dropped silently by the inner merge in _endpoint_ML
        missing = len(set(self.df_all['compound']) - set(self.MF_features['all']['compound']))
        assert not missing, f'{missing} df_all compounds have no MF features'
        print(f"> MF features ({type}) [all]: {self.MF_features['all'].shape[0]} compounds x "
              f"{self.MF_features['all'].shape[1] - 1} features "
              f"(public {len(pub)} cached | internal {len(internal)} recomputed)")

    def _compute_MF(self, smi, type, n_jobs):
        """Compute one feature block from a [compound, smiles] frame ('H237' or 'H236')."""
        if type == 'H237':
            return rdkit_tools.compute_H237_features(smi, n_jobs=n_jobs, v=True)
        if type == 'H236':
            return rdkit_tools.compute_H236_features(smi, v=True)
        raise ValueError(f"unknown feature type {type!r} (use 'H237' or 'H236')")

    def _feats(self, endpoint):
        """Return the endpoint's own feature matrix if built, else the unified MF_features['all'] (from build_MF_features)."""
        return self.MF_features[endpoint] if endpoint in self.MF_features else self.MF_features['all']

    # ---- generic, config-driven builders (used by all endpoints via the aliases below) ----
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

    def _endpoint_ML(self, endpoint):
        """
        -Generic modelling frame for ANY endpoint: optionally winsorize the label at the config cap
         (ADME_ENDPOINTS[endpoint]['label_cap_raw'], raw units -> modelling space via the transform),
         merge the H237 features onto the labels, drop feature NaNs + non-finite labels, and dedup molecules
         by SMILES (prefer the internal row on a clash). Store in self.ML_data[endpoint].
        param str endpoint: endpoint key
        return None: (self.ML_data[endpoint] — compound, smiles, label, source, origin + H237 features)
        """
        ep = self.params.ADME_ENDPOINTS[endpoint]; fwd, _ = _TF[ep['transform']]
        d = self.dfs[endpoint][['compound', 'smiles', 'label', 'source', 'origin']].copy()
        cap = ep.get('label_cap_raw')                                      # optional raw-unit upper cap (e.g. solubility 15000 µM)
        if cap is not None:
            d['label'] = d['label'].clip(upper=fwd(cap))                   # winsorize saturated high values
        d = pd.merge(self._feats(endpoint), d, on='compound').dropna()
        d = d[np.isfinite(d['label'])]                                     # drop non-finite labels (log10/logit of a 0/edge value)
        d = (d.assign(_int=(d['source'] == 'internal'))                    # dedup molecules: prefer the internal row on a SMILES clash
               .sort_values('_int', ascending=False, kind='stable')
               .drop_duplicates('smiles', keep='first').drop(columns='_int'))
        self.ML_data[endpoint] = d.reset_index(drop=True)
        cap_note = f" (cap {cap:g} raw)" if cap is not None else ""
        print(f"> ML_data['{endpoint}']: {self.ML_data[endpoint].shape[0]} rows x {self.ML_data[endpoint].shape[1]} cols{cap_note}")

    def build_ML_data_RF(self, params, k=None):
        """
        -RF modelling frame for one endpoint: the named entry point for _endpoint_ML, so a caller does not
         need exec('data.build_ML_data_'+k+'()'). Winsorizes the label at the config cap, merges the H237
         features onto the labels, drops feature NaNs and non-finite labels, and dedups molecules by SMILES.
        param class params: PARAMS instance (ADME_ENDPOINTS)
        param str k: endpoint key (default self.k)
        return None: (sets self.ML_data[k])
        """
        self.params = params
        self._endpoint_ML(k or self.k)

    def build_ML_data_CP(self, params, k=None, grouping=None, leak='exact', n_splits=5, seed=42):
        """
        -Chemprop analogue of build_ML_data_<ep>: build the two frames the transfer arm needs, from the WIDE
         df_all plus the precomputed descriptastorus DS_ columns of the H237 matrix. cp_pub is the stage-1
         PUBLIC pretrain pool (rows with >=1 grouping task measured, never an internal label). cp_int is the
         stage-2 INTERNAL finetune pool (rows measured for k; the other grouping tasks ride along as masked
         auxiliaries). Public molecules that also exist internally are dropped (leak control). cp_folds come
         from RF when its pickle exists, else from an independent grouped split.
        param class params: PARAMS instance (BEST_CHEMPROP_GROUPINGS, CHEMPROP_TRANSFER)
        param str k: endpoint key to score (default self.k)
        param str grouping: pretrain grouping name (default BEST_CHEMPROP_GROUPINGS[k])
        param str leak: leak-control level for fn.drop_internal_twins ('exact' | 'skeleton' | 'none')
        param int n_splits: fold count, used only when RF's pickle is absent
        param int seed: fold seed, used only when RF's pickle is absent
        return None: (sets self.cp_k, cp_grouping, cp_tasks, cp_ds_cols, cp_pub, cp_int, cp_folds, cp_fold_source)
        """
        self.params = params
        k = k or self.k
        assert 'all' in self.MF_features, 'run build_MF_features(params) first'
        cfg = params.CHEMPROP_TRANSFER
        self.cp_k = k
        self.cp_grouping = grouping or params.BEST_CHEMPROP_GROUPINGS[k]
        # a CLUSTER grouping co-trains the target endpoint with raw Novartis columns from its own parquet;
        # an ENDPOINT grouping uses internal endpoint columns only, so df_all already holds every task
        spec = cfg.get('clusters', {}).get(self.cp_grouping)
        aux = None
        if spec is not None:
            path = os.path.join(getattr(params, 'ADME_CACHE', 'autoresearch/predict_adme'), spec['file'])
            aux = pd.read_parquet(path).drop(columns='_ik', errors='ignore').drop_duplicates('smiles')
            main = [spec['target']]
        else:
            main = list(cfg['groupings'][self.cp_grouping])
        self.cp_tasks = main + ([c for c in aux.columns if c != 'smiles'] if aux is not None else [])
        # chemprop's features are the 200 precomputed descriptastorus columns, not the full H237 matrix
        self.cp_ds_cols = [c for c in self.MF_features['all'].columns if c.startswith('DS_')]
        keep = ['compound', 'smiles', 'source', 'origin'] + main + self.cp_ds_cols
        wide = self.df_all.merge(self.MF_features['all'][['compound'] + self.cp_ds_cols],
                                 on='compound', how='left')[keep]
        # one LEFT merge adds the auxiliary tasks, then INTERNAL rows are blanked again: an internal molecule
        # that also exists in the Novartis frame must not receive its PREDICTED aux values (that is the
        # calibration bias THE LAW warns about), and it keeps the runner's behaviour, where internal aux is NaN
        if aux is not None:
            wide = wide.merge(aux, on='smiles', how='left')
            wide.loc[wide.source == 'internal', [c for c in aux.columns if c != 'smiles']] = np.nan
        # a non-finite label (log10 of 0, logit_pct of 0/100 %) breaks chemprop's target scaler -> mask it as
        # NaN, so the multitask loss ignores that one task and the row keeps its other labels
        n_inf = int(np.isinf(wide[self.cp_tasks].to_numpy(dtype=float)).sum())
        wide[self.cp_tasks] = wide[self.cp_tasks].replace([np.inf, -np.inf], np.nan)
        # stage 1 pool: PUBLIC rows with at least one grouping task measured
        self.cp_pub = wide[(wide.source != 'internal') & wide[self.cp_tasks].notna().any(axis=1)].reset_index(drop=True)
        # stage 2 pool: INTERNAL rows measured for k
        self.cp_int = wide[(wide.source == 'internal') & wide[k].notna()].reset_index(drop=True)
        # leak control: a public molecule that also exists internally must not pretrain the model
        self.cp_pub, _ = fn.drop_internal_twins(self.cp_pub, self.cp_int, level=leak)
        self.cp_folds, self.cp_fold_source = self._cp_folds(params, k, n_splits, seed)
        print(f"> CP_data '{self.cp_grouping}' for '{k}': tasks={self.cp_tasks} | DS_ features={len(self.cp_ds_cols)}")
        print(f"  cp_pub  {len(self.cp_pub):>7} public rows   non-null: "
              + str({t: int(self.cp_pub[t].notna().sum()) for t in self.cp_tasks}))
        print(f"  cp_int  {len(self.cp_int):>7} internal rows non-null: "
              + str({t: int(self.cp_int[t].notna().sum()) for t in self.cp_tasks}))
        print(f"  cp_folds {len(self.cp_folds)} folds (train,test)="
              f"{[(len(a), len(b)) for a, b in self.cp_folds]} | source={self.cp_fold_source}")
        if n_inf:
            print(f"  masked {n_inf} non-finite task value(s) as NaN (unusable label, other tasks kept)")

    def _cp_folds(self, params, k, n_splits=5, seed=42):
        """
        -Fold ids for the chemprop transfer arm. Prefer RF's EXACT folds, recovered from the `fold` column of
         <METRICS_PKL_RF_DIR>/<k>.pkl, so chemprop and RF score the identical partitions and the
         two R2det values compare directly. When that pickle does not exist, build an independent InChIKey-
         grouped split over self.cp_int instead, and say so — those folds are NOT comparable to RF fold by fold.
        param class params: PARAMS instance (METRICS_PKL_RF_DIR, FOLD_GROUP_BY_INCHIKEY)
        param str k: endpoint key
        param int n_splits: fold count for the independent split
        param int seed: KFold seed for the independent split
        return tuple: ([[train_ids, test_ids], ...], source string 'rf:<path>' or 'independent')
        """
        path = os.path.join(params.METRICS_PKL_RF_DIR, f'{k}.pkl')
        if os.path.exists(path):
            print(f"> cp_folds: RF's EXACT folds from {path} — chemprop and RF score the identical partitions")
            with open(path, 'rb') as fh:
                p = pickle.load(fh)[k]['internal_cv_preddf']
            assert 'fold' in p.columns, f'{path}: internal_cv_preddf has no fold column'
            # every RF fold compound must exist in cp_int, else the two models score different sets
            miss = len(set(p.compound) - set(self.cp_int.compound))
            if miss:
                print(f"  CAUTION: {miss} RF fold compounds are absent from cp_int — the sets are NOT identical")
            return ([[list(p.compound[p.fold != f]), list(p.compound[p.fold == f])]
                     for f in sorted(p.fold.unique())], f'rf:{path}')
        # no RF pickle -> independent folds over cp_int, grouped by InChIKey (config FOLD_GROUP_BY_INCHIKEY)
        print(f"> cp_folds: {path} NOT FOUND -> INDEPENDENT {n_splits}-fold split over cp_int (seed {seed}); "
              f"these folds are not comparable to RF fold by fold")
        ik = fn.inchikeys_for(self.cp_int) if getattr(params, 'FOLD_GROUP_BY_INCHIKEY', True) else None
        return self._grouped_folds(n_splits, seed, df=self.cp_int.assign(_ik=ik)), 'independent'

    def get_internal_public_sets(self, k, min_n=1000):
        """
        -Split endpoint k's modelling frame (self.ML_data[k]) into internal vs public sets and rank the
         public origins. Add an InChIKey column, keep the internal rows, keep the public (EXP) rows whose
         InChIKey is NOT an internal twin (drop leaks), and list public origins with more than min_n compounds.
        param str k: endpoint key (index into self.ML_data)
        param int min_n: keep a public origin only when it has more than this many compounds
        return None: (sets self.k, self.d, self.internal, self.pub, self.origins)
        """
        # remember the current endpoint key (used downstream by OUTPUT.assess_predictions)
        self.k = k
        # add an InChIKey column for leak detection between internal and public
        self.d = self.ML_data[k].assign(_ik=fn.smiles_to_inchikeys(self.ML_data[k].smiles))
        # internal rows
        self.internal = self.d[self.d.source == 'internal']
        # public (EXP) rows, minus any InChIKey that also appears internally (drop internal-twin leaks)
        self.pub = self.d[(self.d.source == 'EXP') & ~self.d._ik.isin(set(self.internal._ik.dropna()))]
        # public origins with more than min_n compounds
        self.origins = self.pub.origin.value_counts().loc[lambda s: s > min_n].index.tolist()
        print(f"> {k}: internal={len(self.internal)}, public={len(self.pub)} (origins > {min_n}: {self.origins})")

    def transfer(self, combo, make_model, use_cuml=False):
        """
        -Score one public-origin combo: train on the combo's public compounds, predict the internal set,
         and return the external->internal transfer metrics (mirrors the notebook `transfer`). Uses
         self.d / self.internal / self.pub; `make_model` (from the notebook) builds a fresh champion RF.
        param tuple combo: public origins to train on
        param callable make_model: make_model(use_cuml) -> a fresh estimator
        param bool use_cuml: build the cuRF backend for this fit
        return dict: {combo, n_train, R2_pears, R2_det, RMSE}
        """
        # public compounds for this combo -> train; predict all internal
        tr = self.pub.compound[self.pub.origin.isin(combo)].tolist()
        _, pr = ML_Reg.K_fold_by_defined_IDs(self.d, 'compound', [[tr, self.internal.compound.tolist()]],
                                             model=make_model(use_cuml),
                                             col_to_rm=['compound', 'smiles', 'label', 'source', 'origin', '_ik'], v=False)
        y, p = pr.real_y.values, pr.pred_y.values
        return {'combo': '+'.join(o.split(':')[-1] for o in combo), 'n_train': len(tr),
                'R2_pears': np.corrcoef(y, p)[0, 1] ** 2,
                'R2_det': 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                'RMSE': (((y - p) ** 2).mean()) ** 0.5}

    def _grouped_folds(self, n_splits=5, seed=42, df=None):
        """
        -Build n_splits CV folds over a frame, grouped by InChIKey so every twin of a molecule (same
         _ik, different SMILES) shares a fold — a molecule never appears in both train and test. Keeps the
         seed-determinism: unique groups are KFold-split (shuffled, seeded), then each compound inherits its
         group's fold. Compounds with a missing InChIKey form their own singleton group.
        param int n_splits: number of CV folds
        param int seed: KFold shuffle seed (group-level, so reproducible)
        param dataframe df: frame with compound + _ik columns (default self.internal)
        return list: [[train_ids, test_ids], ...] over the frame's compound ids
        """
        d = self.internal if df is None else df
        # group id = InChIKey, or the compound id itself when the InChIKey is missing (singleton group)
        g = d._ik.where(d._ik.notna(), d.compound)
        comp, grp = d.compound.to_numpy(), g.to_numpy()
        uniq = pd.unique(grp)
        # assign each unique group to a test fold, then map every compound to its group's fold
        fold_of = {}
        for fnum, (_, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(uniq)):
            fold_of.update({uniq[j]: fnum for j in te})
        cfold = np.array([fold_of[x] for x in grp])
        return [[list(comp[cfold != f]), list(comp[cfold == f])] for f in range(n_splits)]

    def select_best_combo_and_update(self, params, k):
        """
        -Select the best public-source combo for endpoint k and build the id splits used by the prediction
         arms. combo = the config BEST_PUBLIC[k] origins that are present in self.origins (falls back to every
         origin when BEST_PUBLIC has no entry). Builds 5-fold CV over internal (public added to TRAIN only)
         and an 80/20 temporal split by SRB id.
        param class params: PARAMS instance (BEST_PUBLIC)
        param str k: endpoint key
        return None: (sets self.combo, self.pub_ids, self.int_ids, self.fold_ids, self.fold_ids_aug,
                      self.train_temp_ids, self.test_temp_ids)
        """
        # best public origins for k from config; keep only those actually present (else all origins)
        l = params.BEST_PUBLIC.get(k)
        self.combo = [o for o in self.origins if o in l] if l else list(self.origins)
        # public ids for the chosen combo + all internal ids
        self.pub_ids = self.pub.compound[self.pub.origin.isin(self.combo)].tolist()
        self.int_ids = self.internal.compound.to_numpy()
        # 5-fold CV over internal; augmented reuses the folds with public added to TRAIN only (never test).
        # Group by InChIKey (config FOLD_GROUP_BY_INCHIKEY, default on) so a molecule's twins never straddle
        # a train/test split (avoids optimistic leakage from near-duplicate internal analogs).
        if getattr(params, 'FOLD_GROUP_BY_INCHIKEY', True):
            self.fold_ids = self._grouped_folds(n_splits=params.RF_SINGLETASK.get('cv_folds', 5),
                                                seed=params.RF_SINGLETASK['seed'])
        else:
            self.fold_ids = [[list(self.int_ids[tr]), list(self.int_ids[te])]
                             for tr, te in KFold(5, shuffle=True, random_state=42).split(self.int_ids)]
        self.fold_ids_aug = [[tr + self.pub_ids, te] for tr, te in self.fold_ids]
        # 80/20 temporal split by SRB id order
        srb = self.internal.compound.str.extract(r'(\d+)')[0].astype(float).to_numpy()
        order = self.internal.compound.to_numpy()[np.argsort(srb)]
        cut = int(len(order) * 0.7) # proportion of compounds going to train -> temporal
        self.train_temp_ids, self.test_temp_ids = list(order[:cut]), list(order[cut:])


# config-driven per-endpoint aliases, kept for the older exec('data.<step>_'+k+'()') callers in python/.
# BOTH steps are fully generic for EVERY endpoint: get_<ep>_data -> _endpoint_dfs (public files +
# transform + filter from config), build_ML_data_<ep> -> _endpoint_ML (label cap from config label_cap_raw).
# New code calls the named entry points instead: build_ML_data_RF(params, k) and build_ML_data_CP(params, k).
for _ep in ('solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb'):
    setattr(DATA, f'get_{_ep}_data',      lambda self, params, _e=_ep: self._endpoint_dfs(params, _e))
    setattr(DATA, f'build_ML_data_{_ep}', lambda self, _e=_ep: self._endpoint_ML(_e))


class OUTPUT():
    """Model machinery + result store: builds the champion RF, runs the prediction arms, and holds the
    per-endpoint metrics and the public-origin combinatorial scan. DATA holds the datasets; OUTPUT the results."""

    # columns dropped before featurizing (id / label / meta), shared by every prediction arm
    COL2RM = ['compound', 'smiles', 'label', 'source', 'origin', '_ik']

    def __init__(self, params):
        self.cfg = params.RF_SINGLETASK            # champion hyperparameters + seed + n_jobs
        self.model = self.make_model(False)        # default sklearn champion RF (K_fold clones it per fold)
        self.metrics_results = {}                  # endpoint -> {<arm>_preddf, <arm>_metrics, ...}
        self.metrics_results_cp = {}               # endpoint -> the same, for the chemprop arms (assess_predictions_cp)
        self.res_origin_powerset = None            # public-origin combinatorial scan (run_origin_powerset)

    def make_model(self, use_cuml=False, n_bins=32):
        """Champion RF: sklearn (default) or RAPIDS cuRF on GPU (use_cuml=True; needs the `rapids` kernel,
        no uq_std/confidence). n_bins is the cuRF speed knob. Mirrors the notebook make_model."""
        if use_cuml:
            from cuml.ensemble import RandomForestRegressor as cuRF
            return cuRF(**self.cfg['champion'], n_bins=n_bins, n_streams=4, random_state=self.cfg['seed'])
        return RandomForestRegressor(**self.cfg['champion'], n_jobs=self.cfg['n_jobs'], random_state=self.cfg['seed'])

    def predict_and_record(self, d, result_df, exp_name='default', ids=None, col_to_rm=None, use_cuml=False):
        """Run one CV/holdout arm: predict via ML_Reg.K_fold_by_defined_IDs (with tree-variance UQ), then
        store the pred_dasss + absolute residual. The confidence columns are added later,
        endpoint-wide, by assess_predictions (calibrated on the augmented_cv arm)."""
        # pick the backend for this arm (cuRF on request, else the default sklearn model)
        mdl = self.make_model(True) if use_cuml else self.model
        # predict the defined id splits (uq=True adds the per-row tree-variance std)
        _, result_df[exp_name + '_preddf'] = ML_Reg.K_fold_by_defined_IDs(
            d, 'compound', ids, model=mdl, col_to_rm=col_to_rm or self.COL2RM, v=False, uq=True)
        # regression metrics for this arm (n_train = full frame size)
        result_df[exp_name + '_metrics'] = ML_Reg.get_reg_metrics_from_preddf(result_df[exp_name + '_preddf'], ntrain=d.shape[0])
        # absolute residual (confidence columns are added later by assess_predictions)
        result_df[exp_name + '_preddf']['residuals'] = abs(result_df[exp_name + '_preddf']['pred_y'] - result_df[exp_name + '_preddf']['real_y'])
        return result_df

    def run_origin_powerset(self, data, use_cuml=False, v=True):
        """
        -Run the public-origin combinatorial scan: score every non-empty subset of data.origins with
         data.transfer (external-public -> internal transfer) and rank by squared Pearson R².
        param DATA data: the endpoint's built sets (origins + d/internal/pub, via get_internal_public_sets)
        param bool use_cuml: build the cuRF backend for every fit
        param bool v: show the powerset progress bar
        return DataFrame: (also stored in self.res_origin_powerset — one row per combo, best R2_pears first)
        """
        # every non-empty subset of the public origins
        combos = [c for j in range(1, len(data.origins) + 1) for c in combinations(data.origins, j)]
        # score each combo (one bar across all combos; the inner K_fold bar is silenced in data.transfer)
        self.res_origin_powerset = (pd.DataFrame(data.transfer(c, self.make_model, use_cuml)
                                                 for c in tqdm(combos, desc='powerset', disable=not v))
                                     .sort_values('R2_pears', ascending=False).reset_index(drop=True))
        return self.res_origin_powerset

    def assess_predictions(self, data, outpath, ml_model='rf'):
        """
        -Compute (or load if outpath already exists) the 6 prediction arms for endpoint data.k and store them
         in self.metrics_results[data.k]. Arms: ext->internal, internal_cv, augmented_cv, internal_temp,
         augmented_temp, ext->temp. Then add the 4 conf_* columns (conf_labelstd / conf_rmse / conf_recal /
         conf_conformal) leakage-controlled: CV arms via leave-one-fold-out (each fold scored by the others),
         single-block arms via a calibration fit on augmented_cv EXCLUDING that arm's test compounds. The
         pooled augmented_cv calibration is stored under '_calibration' for the webapp. On first compute, pickle.
        param DATA data: the endpoint's built sets (k, d, internal, pub_ids, int_ids, fold_ids, ...)
        param str outpath: pickle cache path (load if present, else compute + save)
        param str ml_model: 'rf' (this body) or 'chemprop' (dispatch to assess_predictions_cp)
        return dict: (also stored in self.metrics_results[data.k])
        """
        # the chemprop arms need a pretrain stage and different data, so they live in their own method
        if ml_model == 'chemprop':
            return self.assess_predictions_cp(data, outpath)
        k = data.k
        if not os.path.exists(outpath):
            r = {}
            # external public -> internal
            r = self.predict_and_record(data.d, r, 'ext_->_internal', [[data.pub_ids, data.int_ids]])
            # internal 5-fold CV
            r = self.predict_and_record(data.internal, r, 'internal_cv', data.fold_ids)
            # augmented 5-fold CV (public added to TRAIN only) — this arm calibrates the confidence
            r = self.predict_and_record(data.d, r, 'augmented_cv', data.fold_ids_aug)
            # internal temporal 80/20
            r = self.predict_and_record(data.d, r, 'internal_temp', [[data.train_temp_ids, data.test_temp_ids]])
            # augmented temporal (public added to the temporal TRAIN)
            r = self.predict_and_record(data.d, r, 'augmented_temp', [[data.train_temp_ids + data.pub_ids, data.test_temp_ids]])
            # external public -> temporal hold-out test
            r = self.predict_and_record(data.d, r, 'ext_->_temp', [[data.pub_ids, data.test_temp_ids]])
            # confidence (leakage-controlled). deploy params = pooled augmented_cv, stored for the webapp
            acv = r['augmented_cv_preddf']
            r['_calibration'] = ML_Reg.calibrate_confidence_params(acv)
            # CV arms: leave-one-fold-out — each fold scored by a calibration fit on the OTHER folds
            for a in ['internal_cv', 'augmented_cv']:
                lofo = ML_Reg.apply_confidences_lofo(r[a + '_preddf'])
                r[a + '_preddf'] = lofo if lofo is not None else ML_Reg.apply_confidences(r[a + '_preddf'], r['_calibration'])
            # single-block arms (1 test block, can't LOFO): calibrate on augmented_cv EXCLUDING this arm's
            # test compounds (fully honest); fall back to the pooled deploy calibration if too few rows remain
            for a in ['ext_->_internal', 'internal_temp', 'augmented_temp', 'ext_->_temp']:
                cal_rows = acv[~acv['compound'].isin(set(r[a + '_preddf']['compound']))]
                calib_a = ML_Reg.calibrate_confidence_params(cal_rows) if len(cal_rows) >= 30 else r['_calibration']
                r[a + '_preddf'] = ML_Reg.apply_confidences(r[a + '_preddf'], calib_a)
            self.metrics_results[k] = r
            # dump the whole dict to the endpoint's pickle
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(outpath, 'wb') as _f:
                pickle.dump(self.metrics_results, _f)
            print(f"> saved metrics_results -> {outpath} ({len(self.metrics_results)} endpoint(s): {list(self.metrics_results)})")
        else:
            # load this endpoint's saved metrics
            with open(outpath, 'rb') as _f:
                loaded = pickle.load(_f)
            self.metrics_results[k] = loaded[k] if isinstance(loaded, dict) and k in loaded else loaded
            print(f"> loaded metrics_results['{k}'] <- {outpath}")
        return self.metrics_results[k]

    def _record_cp(self, r, arm, pred_df, ntrain):
        """Store one chemprop arm: the pred_df, its regression metrics and the absolute residual. No conf_*
        columns — a plain `regression` head gives no per-row std, so there is nothing to calibrate."""
        pred_df['residuals'] = abs(pred_df['pred_y'] - pred_df['real_y'])
        r[arm + '_preddf'] = pred_df
        r[arm + '_metrics'] = ML_Reg.get_reg_metrics_from_preddf(pred_df, ntrain=ntrain)
        m = r[arm + '_metrics']
        print(f"  [{arm}] r2det={m['r2det']:.3f} r2={m['r2']:.3f} rmse={m['rmse']:.3f} n_test={len(pred_df)}", flush=True)
        return r

    def assess_predictions_cp(self, data, outpath):
        """
        -Compute (or load if outpath already exists) the 3 chemprop arms for endpoint data.cp_k and store them
         in self.metrics_results_cp[data.cp_k]. The arms mirror the RF ones of the same name, so the two
         models compare directly on data.cp_folds:
           augmented_cv   — public PRETRAIN + frozen-encoder finetune per fold, predict the held-out fold
                            (public enters TRAIN only, exactly like the RF augmented_cv arm).
           internal_cv    — the same folds with NO pretrain: every fold trains from scratch on internal only.
           ext_->_internal— the public-only pretrain checkpoint predicts every internal compound, no finetune.
         The stage-1 checkpoint is built once and shared by the 5 folds and by ext_->_internal; it never sees
         an internal label. Provenance (grouping, tasks, fold source, checkpoint) is stored under '_*' keys.
         No conf_* columns: chemprop's regression head returns no per-row std.
        param DATA data: a DATA with build_ML_data_CP already run (cp_k, cp_int, cp_pub, cp_folds, cp_tasks)
        param str outpath: pickle cache path (load if present, else compute + save)
        return dict: (also stored in self.metrics_results_cp[data.cp_k])
        """
        k = data.cp_k
        if os.path.exists(outpath):
            with open(outpath, 'rb') as f:
                loaded = pickle.load(f)
            self.metrics_results_cp[k] = loaded[k] if isinstance(loaded, dict) and k in loaded else loaded
            print(f"> loaded metrics_results_cp['{k}'] <- {outpath}")
            return self.metrics_results_cp[k]

        cfg = data.params.CHEMPROP_TRANSFER
        # descriptor flags must be identical at pretrain, finetune and predict (v2 stores only the scaler)
        desc = data.cp_ds_cols if cfg['descriptors'] == 'precomputed' else None
        mfeat = cfg['molecule_featurizers'] if cfg['descriptors'] == 'featurizer' else None
        run = os.path.join(cfg['output_dir'], f'{data.cp_grouping}_{k}')
        kw = dict(target=k, targets=data.cp_tasks, descriptor_cols=desc, molecule_featurizers=mfeat,
                  hp=cfg['hp'], epochs=cfg['epochs_finetune'], patience=cfg['patience'],
                  seed=cfg.get('seed', 42), v=True)
        cp_bin = os.path.expanduser(cfg['chemprop_bin'])
        r = {'_grouping': data.cp_grouping, '_tasks': list(data.cp_tasks), '_fold_source': data.cp_fold_source}

        # augmented_cv: public pretrain (once) -> frozen finetune per fold -> predict the held-out fold
        print(f"> [{k}/{data.cp_grouping}] augmented_cv: transfer on {len(data.cp_folds)} folds", flush=True)
        aug = ML_Reg.chemprop_finetune_K_fold_by_defined_IDs(
            data.cp_int, 'compound', data.cp_folds, cp_bin, data.cp_pub,
            epochs_pretrain=cfg['epochs_pretrain'], freeze_encoder=cfg['freeze_encoder'],
            pretrain_dir=os.path.join(cfg['pretrain_dir'], data.cp_grouping),
            workdir=os.path.join(run, 'augmented_cv'), **kw)
        ckpt = aug.attrs['pretrain_checkpoint']
        r['_pretrain_checkpoint'] = ckpt
        r = self._record_cp(r, 'augmented_cv', aug, len(data.cp_int))

        # internal_cv: the same folds with df_pretrain=None, so no public data and no pretrained weights
        print(f"> [{k}] internal_cv: from scratch, internal only", flush=True)
        r = self._record_cp(r, 'internal_cv', ML_Reg.chemprop_finetune_K_fold_by_defined_IDs(
            data.cp_int, 'compound', data.cp_folds, cp_bin, None,
            workdir=os.path.join(run, 'internal_cv'), **kw), len(data.cp_int))

        # ext_->_internal: the public-only checkpoint predicts every internal compound (no finetune, no folds)
        print(f"> [{k}] ext_->_internal: public-only checkpoint on {len(data.cp_int)} internal compounds", flush=True)
        r = self._record_cp(r, 'ext_->_internal', ML_Reg.chemprop_predict_from_checkpoint(
            data.cp_int, 'compound', ckpt, cp_bin, k, targets=data.cp_tasks, descriptor_cols=desc,
            molecule_featurizers=mfeat, workdir=os.path.join(run, 'ext_to_internal')), len(data.cp_pub))

        self.metrics_results_cp[k] = r
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, 'wb') as f:
            pickle.dump(self.metrics_results_cp, f)
        print(f"> saved metrics_results_cp -> {outpath} ({len(self.metrics_results_cp)} endpoint(s): "
              f"{list(self.metrics_results_cp)})")
        return r

    def assess_all_endpoints(self, data, params, endpoints=None, min_n=1000):
        """
        -used in CLI: python/ADME_build_ML.py --assess_RF_all_endpoints
        -Run the full single-task RF assessment for every endpoint: build the ML frame, split internal vs
         public, select the BEST_PUBLIC combo, and compute-or-load the 6 prediction arms. One pickle per
         endpoint is written under params.METRICS_PKL_RF_DIR (<dir>/<k>.pkl); results collect in
         self.metrics_results. Requires data.df_all + data.MF_features['all'] to be built first.
        param DATA data: a DATA with load_combine_dfs + build_MF_features already run
        param class params: PARAMS instance (ADME_ENDPOINTS, BEST_PUBLIC, METRICS_PKL_RF_DIR)
        param list endpoints: endpoints to run (default: all params.ADME_ENDPOINTS)
        param int min_n: public-origin minimum-count filter (get_internal_public_sets)
        return dict: self.metrics_results (endpoint -> arm results)
        """
        eps = endpoints or list(params.ADME_ENDPOINTS)
        for k in eps:
            # build the modelling frame (features from MF_features['all']) + integrity check
            data.build_ML_data_RF(params, k=k)
            assert stats_tools.check_ML_data(data.ML_data[k], extra_meta_cols=('source', 'origin'), verbose=False)
            # split internal vs public + pick the BEST_PUBLIC combo and id splits
            data.get_internal_public_sets(k, min_n=min_n)
            data.select_best_combo_and_update(params, k)
            # compute-or-load the 6 arms -> <METRICS_PKL_RF_DIR>/<k>.pkl
            self.assess_predictions(data, os.path.join(params.METRICS_PKL_RF_DIR, f'{k}.pkl'))
        return self.metrics_results

    def assess_all_endpoints_cp(self, data, params, endpoints=None, leak='exact', n_splits=5, seed=42):
        """
        -used in CLI: python/ADME_build_ML.py --assess_CP_all_endpoints
        -Chemprop twin of assess_all_endpoints: for every endpoint, build the two chemprop pools on RF's exact
         folds and compute-or-load the 3 arms (augmented_cv / internal_cv / ext_->_internal). One pickle per
         endpoint is written under params.METRICS_PKL_CP_DIR (<dir>/<k>.pkl); results collect
         in self.metrics_results_cp. Requires data.df_all + data.MF_features['all'] to be built first.
         CAUTION: this trains 11 chemprop models per endpoint (1 pretrain + 5 transfer folds + 5 scratch
         folds), so run it detached. The pretrain is cached per GROUPING, so endpoints that share a grouping
         (hlm/mlm/rlm, solubility/logd) pretrain once between them.
        param DATA data: a DATA with load_combine_dfs + build_MF_features already run
        param class params: PARAMS instance (ADME_ENDPOINTS, BEST_CHEMPROP_GROUPINGS, CHEMPROP_TRANSFER, METRICS_PKL_CP_DIR)
        param list endpoints: endpoints to run (default: all params.ADME_ENDPOINTS)
        param str leak: leak-control level for build_ML_data_CP ('exact' | 'skeleton' | 'none')
        param int n_splits: fold count, used only for an endpoint whose RF pickle is absent
        param int seed: fold seed, used only for an endpoint whose RF pickle is absent
        return dict: self.metrics_results_cp (endpoint -> arm results)
        """
        eps = endpoints or list(params.ADME_ENDPOINTS)
        out = params.METRICS_PKL_CP_DIR
        for k in eps:
            # build the pretrain/finetune pools + the folds (RF's own whenever its pickle exists)
            data.build_ML_data_CP(params, k=k, leak=leak, n_splits=n_splits, seed=seed)
            # compute-or-load the 3 arms -> <METRICS_PKL_CP_DIR>/<k>.pkl
            self.assess_predictions_cp(data, os.path.join(out, f'{k}.pkl'))
        return self.metrics_results_cp

    def deploy_endpoint(self, data, params, registry, k, dry_run=False):
        """
        -Fit the deployable RF for endpoint k on internal + selected-public (BEST_PUBLIC) rows, calibrate the
         conf_recal confidence on the augmented CV, bundle the model + feature columns + calibration, and
         register a NEW MLTrail model (adme_<k><DEPLOY.experiment_suffix>, features_type from config DEPLOY;
         the champions adme_<k> stay untouched). The full internal+public training set is archived. Requires
         build_ML_data_<k> + get_internal_public_sets + select_best_combo_and_update already run for k.
        param DATA data: the endpoint's built sets (d, int_ids, pub_ids, combo, fold_ids_aug)
        param class params: PARAMS instance (ADME_ENDPOINTS, DEPLOY)
        param registry: MLTrail Registry (None with dry_run -> fit + calibrate but do not register)
        param str k: endpoint key
        param bool dry_run: skip MLTrail registration (returns the summary only)
        return dict: {endpoint, model_id, n_train, sources, calibration, cv_rmse, cv_r2}
        """
        dep, ep = params.DEPLOY, params.ADME_ENDPOINTS[k]
        # H237 feature columns of the modelling frame (drop id/label/meta)
        feats = [c for c in data.d.columns if c not in self.COL2RM]
        # augmented CV (public in TRAIN only, tree-variance UQ) -> conf_recal calibration params
        _, cv = ML_Reg.K_fold_by_defined_IDs(data.d, 'compound', data.fold_ids_aug,
                                             model=self.make_model(False), col_to_rm=self.COL2RM, v=False, uq=True)
        cal = ML_Reg.calibrate_confidence_params(cv)
        calibration = {kk: float(cal[kk]) for kk in ('rmse_cv', 'recal_a', 'recal_b', 'label_std')}
        cvm = ML_Reg.get_reg_metrics_from_preddf(cv, ntrain=len(cv))
        # fit the deployable model on ALL internal + selected public (no held-out fold)
        train = data.d[data.d.compound.isin(list(data.int_ids) + data.pub_ids)]
        rf = self.make_model(False).fit(train[feats], train['label'])
        # bundle: model + trained columns + the conf_recal calibration (self-contained for the webapp)
        bundle = {'model': rf, 'feature_cols': feats, 'endpoint': k, 'features': dep['features_type'],
                  'sources': ['internal'] + list(data.combo), 'n_train': int(len(train)),
                  'transform': ep['transform'], 'unit': ep.get('unit', ''),
                  'calibration': calibration, 'sklearn_ver': __import__('sklearn').__version__}
        summary = {'endpoint': k, 'model_id': None, 'n_train': int(len(train)),
                   'sources': bundle['sources'], 'calibration': calibration,
                   'cv_rmse': cvm.get('rmse'), 'cv_r2': cvm.get('r2')}
        if dry_run or registry is None:
            return summary
        # register a NEW model; mirror the recal scalars into metrics for details()/trail without an artifact load
        name = f"adme_{k}{dep.get('experiment_suffix', '_h237')}"
        metrics = {'rmse_cv': calibration['rmse_cv'], 'r2_cv': cvm.get('r2'), 'rho_cv': cvm.get('spearman_rho'),
                   'recal_a': calibration['recal_a'], 'recal_b': calibration['recal_b']}
        comment = (f"deployed RF on {dep['features_type']}; internal + BEST_PUBLIC ({'+'.join(bundle['sources'])}); "
                   f"fit on {len(train)} rows; conf_recal calibrated on augmented CV; full trainset archived.")
        summary['model_id'] = int(registry.add(
            model_id=None, model=bundle, experiment_name=name,
            experiment_measure=ep['col'].split('_', 1)[1], unit=ep.get('unit', ''),
            model_type='single_task_regression', framework='sklearn', features_type=dep['features_type'],
            training_set=train[['compound', 'smiles', 'label']], smiles_column='smiles',
            compound_id_column='compound', label_column='label', metrics=metrics, comment=comment))
        return summary

    def deploy_all_endpoints(self, data, params, endpoints=None, min_n=1000, dry_run=False):
        """
        -Fit + deploy the RF for every endpoint: build the ML frame, split internal vs public, select the
         BEST_PUBLIC combo, then deploy_endpoint (fit on internal+public, calibrate conf_recal, register a new
         MLTrail model). Requires data.df_all + data.MF_features['all'] to be built first.
        param DATA data: a DATA with load_combine_dfs + build_MF_features already run
        param class params: PARAMS instance (ADME_ENDPOINTS, BEST_PUBLIC, DEPLOY)
        param list endpoints: endpoints to deploy (default: all params.ADME_ENDPOINTS)
        param int min_n: public-origin minimum-count filter (get_internal_public_sets)
        param bool dry_run: fit + calibrate but do not register to MLTrail
        return list: one summary dict per endpoint
        """
        registry = None if dry_run else __import__('mltrail').Registry.from_default()
        summaries = []
        for k in (endpoints or list(params.ADME_ENDPOINTS)):
            # build the modelling frame (features from MF_features['all']) + integrity check
            data.build_ML_data_RF(params, k=k)
            assert stats_tools.check_ML_data(data.ML_data[k], extra_meta_cols=('source', 'origin'), verbose=False)
            # split internal vs public + pick the BEST_PUBLIC combo and id splits
            data.get_internal_public_sets(k, min_n=min_n)
            data.select_best_combo_and_update(params, k)
            # fit + calibrate + register the deployable model
            s = self.deploy_endpoint(data, params, registry, k, dry_run=dry_run)
            summaries.append(s)
            c = s['calibration']
            print(f"> deploy {k}: id={s['model_id']} n_train={s['n_train']} sources={s['sources']} "
                  f"cv_rmse={s['cv_rmse']:.3g} recal(a,b)=({c['recal_a']:.3g},{c['recal_b']:.3g})")
        return summaries


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build/evaluate the ADME single-task ML models.")
    ap.add_argument('--config', default='config/config.yaml', help="path to the YAML config")
    ap.add_argument('--overwrite', action='store_true', help="re-pull df_all from CDD Vault and re-cache")
    ap.add_argument('--assess_RF_all_endpoints', action='store_true',
                    help="assess the champion RF on every endpoint -> <METRICS_PKL_RF_DIR>/<k>.pkl")
    ap.add_argument('--assess_CP_all_endpoints', action='store_true',
                    help="assess the chemprop transfer arms on every endpoint -> <METRICS_PKL_CP_DIR>/<k>.pkl")
    ap.add_argument('--leak', default='exact', choices=['exact', 'skeleton', 'none'],
                    help="chemprop leak control: drop public molecules that also exist internally")
    ap.add_argument('--deploy_RF_all_endpoints', action='store_true',
                    help="fit + deploy the RF (internal + BEST_PUBLIC) for every endpoint and register to MLTrail")
    ap.add_argument('--dry_run', action='store_true', help="deploy: fit + calibrate but do not register to MLTrail")
    ap.add_argument('--endpoints', default=None, help="comma-separated subset for --assess/--deploy")
    ap.add_argument('--min_n', type=int, default=1000, help="public-origin minimum-count filter")
    args = ap.parse_args()

    ## params:
    params = PARAMS(args.config)
    params.load_params()

    ## data:
    data = DATA()
    data.load_df_internal_exp_all(params, overwrite=args.overwrite)

    if args.assess_RF_all_endpoints or args.assess_CP_all_endpoints or args.deploy_RF_all_endpoints:
        # build the unified dataset + all-compound H237 features (shared by assess + deploy)
        data.load_combine_dfs(params)
        data.build_MF_features(params)
        output = OUTPUT(params)
        eps = [e.strip() for e in args.endpoints.split(',')] if args.endpoints else None

    if args.assess_RF_all_endpoints:
        # assess the RF on every endpoint -> <METRICS_PKL_RF_DIR>/<k>.pkl
        output.assess_all_endpoints(data, params, endpoints=eps, min_n=args.min_n)
        print(f"> done: assessed {len(output.metrics_results)} endpoint(s) -> {params.METRICS_PKL_RF_DIR}")

    if args.assess_CP_all_endpoints:
        # assess the chemprop transfer arms on every endpoint -> <METRICS_PKL_CP_DIR>/<k>.pkl
        output.assess_all_endpoints_cp(data, params, endpoints=eps, leak=args.leak)
        print(f"> done: assessed {len(output.metrics_results_cp)} endpoint(s) -> "
              f"{params.METRICS_PKL_CP_DIR}")

    if args.deploy_RF_all_endpoints:
        # fit + deploy the RF (internal + BEST_PUBLIC) for every endpoint and register to MLTrail
        summaries = output.deploy_all_endpoints(data, params, endpoints=eps, min_n=args.min_n, dry_run=args.dry_run)
        ids = [s['model_id'] for s in summaries]
        print(f"> done: deployed {len(summaries)} endpoint(s){' (dry_run)' if args.dry_run else ''} -> MLTrail ids {ids}")
