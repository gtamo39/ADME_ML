"""Systematic single-task RandomForest ADME runner (PARAMS / DATA / OUTPUT / MAIN).

For each of the 8 in-house ADME endpoints:
  1. Evaluate the champion RF (H236 features) in 4 arms and save each pred_df to
     output/predictions_runs/<endpoint>/:
         internal_temporal   augmented_temporal   internal_cv   augmented_cv
     - temporal  = LOCAL per-endpoint split (newest 30% of THAT endpoint's compounds -> test).
     - 5-fold CV = interpolation; folds on the INTERNAL compounds only, public always in TRAIN.
     - augmented = internal + the public sources vetted for that endpoint (config-driven).
  2. Fit the deployable model on internal + augmented and register it to MLTrail (features_type=H236,
     so MLTrail re-derives H236 from SMILES at predict time), archiving the full training set.

Also writes output/predictions_runs/summary.{csv,parquet} + manifest.json. H236 features are cached
once under output/features (compute-once, reload-on-rerun). Aggregate stdout only — never SMILES
or per-compound values.

Structure (mirrors python/Px_interface.py): PARAMS loads config -> attributes; DATA methods take
params and store on self; OUTPUT methods take (data, params); MAIN wires params -> data -> output.
Notebook use: `from python.run_RF_SingleTask_systematic import PARAMS, DATA, OUTPUT` and step through.

Run (env `ML`):  python python/run_RF_SingleTask_systematic.py [--endpoints logd,mdck] [--no-mltrail]
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/Scripts'))
import numpy as np, pandas as pd, yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import stats
import ML_Reg as ML_Reg
import Statistics_tools as stats_tools
import Rdkit_tools as rdkit_tools

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'                 # pre-built target/feature parquets (Phase 1)
FEATDIR = ROOT / 'output/features'                         # script-owned feature cache (compute once, reload)
CONFIG = ROOT / 'config/config.yaml'
PUBLIC_SRC = {'EXP': 'public_{ep}.parquet', 'NVS': 'public_novartis_{ep}.parquet', 'ADM': 'public_admetlab_{ep}.parquet'}
ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']


def _metrics(pred):
    """(in-house Pearson r2, Spearman) for a pred_df; (None, None) if too small / constant."""
    if pred is None or len(pred) < 3 or pred['real_y'].nunique() < 2:
        return None, None
    return (round(float(stats_tools.rsquared(pred['real_y'], pred['pred_y'])), 3),
            round(float(stats.spearmanr(pred['real_y'], pred['pred_y']).statistic), 3))


# ============================================================ CLASSES
class PARAMS:
    """Reads config/config.yaml and exposes every RF_SINGLETASK key as an attribute (params.seed, ...)."""
    def __init__(self, config_path=CONFIG):
        self.load_params(config_path)

    def load_params(self, config_path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        for k, v in cfg['RF_SINGLETASK'].items():
            setattr(self, k, v)
        self.endpoint_cfg = cfg['ADME_ENDPOINTS']              # per-endpoint col/transform/unit
        self.endpoints = list(ENDPOINTS)
        self.rf_kw = dict(self.champion, n_jobs=self.n_jobs, random_state=self.seed)
        return self


class DATA:
    """Loads the in-house targets + H236 features (cached under output/features) and serves per-endpoint
    frames. Methods take `params`, store on `self`, return None; helpers derive per-endpoint views."""

    def load_features(self, params):
        """Compute the internal H236 (+DS if MFDS) features ONCE and cache under output/features/;
        reloaded on rerun (never recomputed after a failure). Sets self.mf, self.feats."""
        FEATDIR.mkdir(parents=True, exist_ok=True)
        h236_cache = FEATDIR / 'internal_H236.parquet'
        if h236_cache.exists():
            mf = pd.read_parquet(h236_cache)
        else:
            src = CACHE / 'internal_MF.parquet'
            if src.exists():                                   # reuse precomputed cache (identical featurizer)
                mf = pd.read_parquet(src).drop_duplicates('compound')
            else:                                              # else featurize from SMILES (bit-identical H236)
                tgt_sm = pd.read_parquet(CACHE / 'internal_targets.parquet')[['compound', 'smiles']]
                mf = rdkit_tools.compute_H236_features(tgt_sm, v=False)
            mf.to_parquet(h236_cache, index=False)
            print(f'  computed+cached internal H236 -> {h236_cache} {mf.shape}', flush=True)
        self.feats = [c for c in mf.columns if c != 'compound']
        if params.features_type == 'MFDS':                     # optional Descriptastorus (needs descriptastorus)
            ds_cache = FEATDIR / 'internal_DS.parquet'
            ds = pd.read_parquet(ds_cache) if ds_cache.exists() else pd.read_parquet(CACHE / 'internal_DS.parquet').drop_duplicates('compound')
            if not ds_cache.exists():
                ds.to_parquet(ds_cache, index=False)
            self.feats += [c for c in ds.columns if c != 'compound']
            mf = mf.merge(ds, on='compound')
        self.mf = mf

    def load_internal(self, params):
        """Merge in-house targets with the H236 features. Sets self.internal (needs load_features first)."""
        self.internal = pd.read_parquet(CACHE / 'internal_targets.parquet').merge(self.mf, on='compound')

    def load_all(self, params):
        self.load_features(params)
        self.load_internal(params)
        return self

    # --- per-endpoint views ---
    def avail_sources(self, ep):
        return [s for s in PUBLIC_SRC if (CACHE / PUBLIC_SRC[s].format(ep=ep)).exists()]

    def augmented_sources(self, ep, params):
        """Public sources for the 'augmented' model: config override per endpoint (e.g. solubility=[EXP],
        since ADMETlab PREDICTED solubility hurts), else all available parquets."""
        override = (params.augmented_sources or {}).get(ep)
        srcs = override if override is not None else self.avail_sources(ep)
        return [s for s in srcs if (CACHE / PUBLIC_SRC[s].format(ep=ep)).exists()]

    def internal_ep(self, ep):
        """Internal rows with endpoint measured: compound, smiles, label, features."""
        return (self.internal[['compound', 'smiles', ep]].dropna(subset=[ep]).rename(columns={ep: 'label'})
                        .merge(self.internal[['compound'] + self.feats], on='compound'))

    def pooled(self, ep, sources):
        """(internal + chosen public rows), the internal-only frame, and the list of public frames."""
        d = self.internal_ep(ep)
        pub = [pd.read_parquet(CACHE / PUBLIC_SRC[s].format(ep=ep))[['compound', 'smiles', 'value'] + self.feats]
                 .rename(columns={'value': 'label'}) for s in sources]
        return pd.concat([d] + pub, ignore_index=True), d, pub

    def temporal_test_ids(self, ep):
        """LOCAL temporal test = newest 30% of the compounds measured for THIS endpoint (by SRB id).
        Per-endpoint (not global): keeps ~70% of each endpoint's data in train."""
        d = self.internal.dropna(subset=[ep])
        srb = d['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
        return set(d['compound'].to_numpy()[np.argsort(srb)[int(len(d) * 0.7):]])


class OUTPUT:
    """Evaluates each endpoint (4 arms), saves pred_dfs, fits + registers the deployable model, and
    writes the summary + manifest. Methods take (data, params)."""

    def eval_temporal(self, data, params, ep, sources):
        ML, d, _ = data.pooled(ep, sources)
        test = data.temporal_test_ids(ep)                      # local: newest 30% of THIS endpoint
        test_ids = d.loc[d['compound'].isin(test), 'compound'].tolist()
        if len(test_ids) < 3:
            return None
        ids = [[ML.loc[~ML['compound'].isin(test), 'compound'].tolist(), test_ids]]
        _, pred = ML_Reg.K_fold_by_defined_IDs(ML, ID='compound', ID_sets=ids,
            model=RandomForestRegressor(**params.rf_kw), col_to_rm=['compound', 'smiles', 'label'], v=False)
        return pred

    def eval_cv(self, data, params, ep, sources):
        ML, d, pub = data.pooled(ep, sources)
        pub_ids = pd.concat(pub)['compound'].tolist() if pub else []
        ids = d['compound'].to_numpy()
        kf = KFold(n_splits=params.cv_folds, shuffle=True, random_state=params.seed)
        ID_sets = [[list(ids[tr]) + pub_ids, list(ids[te])] for tr, te in kf.split(ids)]   # test = internal only
        _, pred = ML_Reg.K_fold_by_defined_IDs(ML, ID='compound', ID_sets=ID_sets,
            model=RandomForestRegressor(**params.rf_kw), col_to_rm=['compound', 'smiles', 'label'], v=False)
        return pred

    def eval_public_only(self, data, params, ep, sources):
        """Train on PUBLIC rows only (zero internal in train), predict ALL internal compounds
        (external validation / domain-transfer baseline). test = every internal compound with ep measured."""
        ML, d, pub = data.pooled(ep, sources)
        if not pub or len(d) < 3:
            return None
        ids = [[pd.concat(pub)['compound'].tolist(), d['compound'].tolist()]]   # train=public, test=all internal
        _, pred = ML_Reg.K_fold_by_defined_IDs(ML, ID='compound', ID_sets=ids,
            model=RandomForestRegressor(**params.rf_kw), col_to_rm=['compound', 'smiles', 'label'], v=False)
        return pred

    def run_public_only(self, data, params):
        """Public-only -> internal for each requested endpoint; save pred_public_only.parquet + summary."""
        rows = []
        for ep in params.endpoints:
            aug = data.augmented_sources(ep, params)
            pred = self.eval_public_only(data, params, ep, aug)
            if pred is not None:
                epdir = ROOT / params.output_dir / ep; epdir.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(epdir / 'pred_public_only.parquet', index=False)
            r2, rho = _metrics(pred)
            rows.append({'endpoint': ep, 'sources': '+'.join(aug), 'publiconly_r2': r2,
                         'publiconly_rho': rho, 'publiconly_n': 0 if pred is None else len(pred)})
            print(f'  {ep:11} PUBLIC->INT  r2={r2} rho={rho} n={rows[-1]["publiconly_n"]}', flush=True)
        out_dir = ROOT / params.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame(rows); prev = out_dir / 'summary_public_only.csv'
        if prev.exists():
            old = pd.read_csv(prev).pipe(lambda o: o[~o['endpoint'].isin(long['endpoint'])])
            long = pd.concat([old, long], ignore_index=True) \
                     .sort_values('endpoint', key=lambda s: s.map({e: i for i, e in enumerate(ENDPOINTS)})).reset_index(drop=True)
        long.to_csv(prev, index=False)
        print(f'\n> wrote {prev}', flush=True)
        print(long.to_string(index=False), flush=True)

    @staticmethod
    def modelling_unit(cfg_ep):
        t, u = cfg_ep['transform'], cfg_ep['unit']
        return {'identity': u, 'log10': f'log10({u})', 'logit_pct': 'logit(fraction_unbound)'}.get(t, f'{t}({u})')

    def register(self, data, params, registry, ep, rf, sources, train_df, metrics, comment):
        cfg_ep = params.endpoint_cfg[ep]
        name = f'adme_{ep}'
        listing = registry.list()
        hit = listing.loc[listing['experiment_name'] == name, 'id'] if len(listing) else pd.Series([], dtype=int)
        bundle = {'model': rf, 'feature_cols': data.feats, 'endpoint': ep, 'features': params.features_type,
                  'sources': ['internal'] + list(sources), 'n_train': int(len(train_df)),
                  'transform': cfg_ep['transform'], 'unit': self.modelling_unit(cfg_ep),
                  'sklearn_ver': __import__('sklearn').__version__}
        return registry.add(model_id=int(hit.iloc[0]) if len(hit) else None, model=bundle,
            experiment_name=name, experiment_measure=cfg_ep['col'].split('_', 1)[1], unit=self.modelling_unit(cfg_ep),
            model_type='single_task_regression', framework='sklearn', features_type='H236',
            training_set=train_df, smiles_column='smiles', compound_id_column='compound', label_column='label',
            metrics=metrics, comment=comment)

    def deploy_sanity(self, registry, model_id):
        """Predict on PUBLIC reference SMILES to confirm the H236 featurizer aligns end-to-end."""
        ref = pd.DataFrame({'compound': ['ethanol', 'benzene', 'aspirin'],
                            'smiles': ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']})
        out = registry.predict(model_id, ref, smiles_column='smiles', compound_id='compound')
        col = 'prediction' if 'prediction' in out.columns else out.select_dtypes('number').columns[-1]
        return int(out[col].notna().sum()), len(ref)

    def run_endpoint(self, data, params, ep, registry, resume=False):
        """Evaluate the 4 arms (save pred_dfs), then fit + register the deployable augmented model.
        resume=True: if the 4 pred_dfs already exist, reload them and skip the re-fits; and skip MLTrail
        registration if `adme_<ep>` is already in the registry — so a restart after a crash picks up where
        it stopped instead of redoing completed endpoints."""
        aug = data.augmented_sources(ep, params)
        epdir = ROOT / params.output_dir / ep; epdir.mkdir(parents=True, exist_ok=True)
        arm_names = ['internal_temporal', 'augmented_temporal', 'internal_cv', 'augmented_cv']
        cached = resume and all((epdir / f'pred_{a}.parquet').exists() for a in arm_names)
        if cached:
            arms = {a: pd.read_parquet(epdir / f'pred_{a}.parquet') for a in arm_names}
            print(f'  {ep:11} [resume] reloaded 4 cached pred_dfs — skipped re-fit', flush=True)
        else:
            arms = {'internal_temporal': self.eval_temporal(data, params, ep, []),
                    'augmented_temporal': self.eval_temporal(data, params, ep, aug),
                    'internal_cv': self.eval_cv(data, params, ep, []),
                    'augmented_cv': self.eval_cv(data, params, ep, aug)}
        row = {'endpoint': ep, 'sources': '+'.join(aug)}
        for arm in arm_names:
            pred = arms[arm]
            if not cached and pred is not None:
                pred.to_parquet(epdir / f'pred_{arm}.parquet', index=False)
            r2, rho = _metrics(pred)
            row[f'{arm}_r2'], row[f'{arm}_rho'], row[f'{arm}_n'] = r2, rho, (0 if pred is None else len(pred))
            print(f'  {ep:11} {arm:20} r2={r2} rho={rho} n={row[f"{arm}_n"]}', flush=True)
        row['deployed'] = 'augmented'                          # always deploy internal + augmented

        if registry is not None:
            listing = registry.list()
            if resume and len(listing) and (listing['experiment_name'] == f'adme_{ep}').any():
                print(f'  {ep:11} [resume] already in MLTrail — skipped registration', flush=True)
            else:
                ML, d, _ = data.pooled(ep, aug)
                rf = RandomForestRegressor(**params.rf_kw).fit(ML[data.feats], ML['label'])
                metrics = {k: v for k, v in {'r2_temporal': row['augmented_temporal_r2'], 'rho_temporal': row['augmented_temporal_rho'],
                                             'r2_cv': row['augmented_cv_r2'], 'rho_cv': row['augmented_cv_rho']}.items() if v is not None}
                comment = (f'champion RF on H236; deployed = internal + augmented (sources: {"+".join(["internal"]+aug)}); '
                           f'fit on {len(ML)} rows; full internal+public trainset archived (reproducible in isolation). '
                           f'Verdict note: Chemprop wins solubility/mlm/hlm — this is the single-task RF baseline.')
                train_df = (ML[['compound', 'smiles', 'label']] if params.archive_public_in_trainset
                            else d[['compound', 'smiles', 'label']])
                mid = self.register(data, params, registry, ep, rf, aug, train_df, metrics, comment)
                ok, tot = self.deploy_sanity(registry, mid)
                print(f'  {ep:11} -> MLTrail id={mid} sources={["internal"]+aug} n_train={len(ML)} '
                      f'metrics={metrics} | sanity {ok}/{tot} non-null', flush=True)
        return row

    def evaluate_all(self, data, params, registry, resume=False):
        return [self.run_endpoint(data, params, ep, registry, resume) for ep in params.endpoints]

    def write_outputs(self, data, params, summary):
        out_dir = ROOT / params.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
        sdf = pd.DataFrame(summary)
        prev = out_dir / 'summary.csv'                          # partial run: merge into the existing table, don't clobber
        if prev.exists() and set(params.endpoints) != set(ENDPOINTS):
            keep = pd.read_csv(prev).pipe(lambda d: d[~d['endpoint'].isin(sdf['endpoint'])])
            sdf = pd.concat([keep, sdf], ignore_index=True)
            sdf = sdf.sort_values('endpoint', key=lambda s: s.map({e: i for i, e in enumerate(ENDPOINTS)})).reset_index(drop=True)
        sdf.to_csv(out_dir / 'summary.csv', index=False)
        sdf.to_parquet(out_dir / 'summary.parquet', index=False)
        try:
            commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD']).decode().strip()
        except Exception:
            commit = None
        (out_dir / 'manifest.json').write_text(json.dumps(
            {'timestamp': datetime.now().isoformat(timespec='seconds'), 'git_commit': commit,
             'features_type': params.features_type, 'n_features': len(data.feats), 'seed': params.seed,
             'cv_folds': params.cv_folds, 'champion': params.champion, 'n_internal': int(len(data.internal)),
             'temporal_split': 'local_per_endpoint_70_30', 'archive_public_in_trainset': params.archive_public_in_trainset,
             'endpoints': params.endpoints, 'augmented_sources': {ep: data.augmented_sources(ep, params) for ep in params.endpoints},
             'temporal_test_n': {ep: len(data.temporal_test_ids(ep)) for ep in params.endpoints}}, indent=2))
        print(f'\n> wrote {out_dir}/summary.csv + summary.parquet + manifest.json', flush=True)
        print(sdf[['endpoint', 'sources', 'internal_temporal_r2', 'augmented_temporal_r2',
                   'internal_cv_r2', 'augmented_cv_r2', 'deployed']].to_string(index=False), flush=True)


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', default=','.join(ENDPOINTS), help='comma-separated subset')
    ap.add_argument('--no-mltrail', action='store_true', help='skip MLTrail registration')
    ap.add_argument('--resume', action='store_true',
                    help='skip endpoints whose 4 pred_dfs already exist (and MLTrail entry) — restart after a crash')
    ap.add_argument('--public-only', dest='public_only', action='store_true',
                    help='train on PUBLIC data only and predict ALL internal (external-validation baseline); no 4-arm run, no MLTrail')
    args = ap.parse_args()

    params = PARAMS(CONFIG)
    params.endpoints = [e for e in args.endpoints.split(',') if e in ENDPOINTS]
    data = DATA().load_all(params)
    print(f'> {len(data.internal)} internal cmpd | {len(data.feats)} {params.features_type} feats | '
          f'LOCAL per-endpoint temporal split (newest 30%) | endpoints={params.endpoints}', flush=True)

    if args.public_only:
        OUTPUT().run_public_only(data, params)
        return

    registry = None
    if not args.no_mltrail and params.register_mltrail:
        from mltrail import Registry
        registry = Registry.from_default()

    output = OUTPUT()
    summary = output.evaluate_all(data, params, registry, resume=args.resume)
    output.write_outputs(data, params, summary)


if __name__ == '__main__':
    main()
