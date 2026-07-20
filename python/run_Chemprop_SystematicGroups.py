"""Systematic Chemprop grouping run — the multitask counterpart of run_RF_SingleTask_systematic.py.

Trains ONE Chemprop D-MPNN per endpoint grouping (augmented+internal, TEMPORAL only — no CV, no
internal-only: too little data for a DNN) and scores each member endpoint on the SAME local
per-endpoint temporal test as the RF run, so RF and Chemprop are directly comparable and every
endpoint is predicted by every grouping that contains it (e.g. solubility by both {sol,logd} and all-8).

Groupings (from autoresearch validation): {sol,logd}, {hlm,mlm,rlm}, all-8.
Augmented sources: the SAME config-vetted per-endpoint policy as RF (RF_SINGLETASK.augmented_sources;
solubility=EXP only). Hyperparameters: read from the HPO best_config.json at run time.

Leakage-safe multitask split: a model shares one graph per compound across its endpoints, so for each
grouping the TEST is the UNION of its members' local-temporal tests (held out entirely); each endpoint
is then scored on ITS OWN newest-30% subset (== the RF test). Train = the rest + public; 10% of train
carved to val for early stopping.

Structure mirrors run_RF_SingleTask_systematic.py: PARAMS / DATA / OUTPUT / MAIN.
Train/eval runs in the `chemprop` env (GPU). MLTrail registration is a separate `ML`-env `--register`
step and is ARCHIVE-ONLY (MLTrail v1 cannot predict chemprop — backend stub).

Run temporal (env `chemprop`):  python python/run_Chemprop_SystematicGroups.py
Run 5-fold CV (env `chemprop`): python python/run_Chemprop_SystematicGroups.py --cv --groupings clearance all8
Register (env `ML`):            python python/run_Chemprop_SystematicGroups.py --register

--cv scores each endpoint by pooled out-of-fold prediction over a shared compound-level K-fold split
(public always in train), giving a CV r2 directly comparable to the RF run's internal_cv_r2 / augmented_cv_r2.
"""
from __future__ import annotations
import os, sys, json, shutil, argparse, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'autoresearch/predict_adme'
CONFIG = ROOT / 'config/config.yaml'
MODELS_DIR = ROOT / 'output/chemprop_models'      # persistent save of trained grouping checkpoints
sys.path.insert(0, str(CACHE))
import run_chemprop_multitask as cp          # reuse CHEMPROP path, _run_quiet, _wide_public, SEED
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.expanduser('~/Scripts'))
from Statistics_tools import rsquared as _rsq

ENDPOINTS = ['solubility', 'logd', 'hlm', 'mlm', 'rlm', 'caco2', 'mdck', 'ppb']
PUBLIC_SRC = {'EXP': 'public_{ep}.parquet', 'NVS': 'public_novartis_{ep}.parquet', 'ADM': 'public_admetlab_{ep}.parquet'}


# ============================================================ CLASSES
class PARAMS:
    """Reads CHEMPROP_SYSTEMATIC from config; augmented-source policy is shared with RF_SINGLETASK."""
    def __init__(self, config_path=CONFIG):
        self.load_params(config_path)

    def load_params(self, config_path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        for k, v in cfg['CHEMPROP_SYSTEMATIC'].items():
            setattr(self, k, v)
        self.augmented_sources_override = (cfg['RF_SINGLETASK'].get('augmented_sources') or {})  # same policy as RF
        self.endpoint_cfg = cfg['ADME_ENDPOINTS']
        self.seed = cfg['RF_SINGLETASK']['seed']
        self.hp = self._load_hyperparams()
        return self

    def _load_hyperparams(self):
        """HPO best architecture -> chemprop train flags. {} (stock defaults) if the file is absent yet."""
        p = ROOT / self.hpopt_config
        if not p.exists():
            print(f'  [warn] {p} not found -> using Chemprop stock defaults (run the HPO first for tuned params)', flush=True)
            return {}
        best = json.loads(p.read_text())['best_params']
        return best


class DATA:
    def load_internal(self, params):
        """In-house targets (compound, smiles, 8 endpoints)."""
        self.tgt = pd.read_parquet(CACHE / 'internal_targets.parquet').reset_index(drop=True)

    def load_all(self, params):
        self.load_internal(params)
        return self

    def augmented_sources(self, ep, params):
        """Config-vetted public sources for ep (same as RF): override else all available parquets."""
        override = params.augmented_sources_override.get(ep)
        srcs = override if override is not None else list(PUBLIC_SRC)
        return [s for s in srcs if (CACHE / PUBLIC_SRC[s].format(ep=ep)).exists()]

    def temporal_test_ids(self, ep):
        """LOCAL temporal test = newest 30% of ep's measured compounds (identical to the RF run)."""
        d = self.tgt.dropna(subset=[ep])
        srb = d['compound'].str.extract(r'(\d+)')[0].astype(float).to_numpy()
        return set(d['compound'].to_numpy()[np.argsort(srb)[int(len(d) * 0.7):]])

    def cv_compound_ids(self, endpoints):
        """Internal compounds with >=1 of the grouping's endpoints measured (the CV pool). One shared
        compound-level fold assignment across the grouping (a compound's graph can't split train/test)."""
        return self.tgt.dropna(subset=endpoints, how='all')['compound'].to_numpy()


class OUTPUT:
    def _public_rows(self, data, params, endpoints):
        """Public training rows (splits='train') per the config-vetted policy: EXP per-endpoint
        single-target rows, NVS/ADM pseudo-labels merged wide. Shared by build_combined/build_public_only."""
        rows = []
        for ep in [e for e in endpoints if 'EXP' in data.augmented_sources(e, params)]:
            r = pd.read_parquet(CACHE / PUBLIC_SRC['EXP'].format(ep=ep), columns=['smiles', 'value']).rename(columns={'value': ep})
            r['splits'] = 'train'; rows.append(r)
        for pref, key in [('public_novartis', 'NVS'), ('public_admetlab', 'ADM')]:
            eps = [e for e in endpoints if key in data.augmented_sources(e, params)]
            w = cp._wide_public(pref, eps) if eps else None
            if w is not None:
                w['splits'] = 'train'; rows.append(w)
        return rows

    def build_combined(self, data, params, endpoints, rng, test_ids=None):
        """Write the combined multitask CSV for one grouping and return (csv_path, test_truth).
        Split: TEST = `test_ids` if given (a CV fold), else the union of members' local-temporal tests;
        rest = train; 10% of train -> val. Public rows (splits=train) added per the config-vetted
        policy: EXP per-endpoint, NVS/ADM wide. Public is ALWAYS in train (never in test), as with RF."""
        union_test = set(test_ids) if test_ids is not None else set().union(*(data.temporal_test_ids(ep) for ep in endpoints))
        tgt = data.tgt
        s = np.where(tgt['compound'].isin(union_test), 'test', 'train').astype(object)
        train_idx = np.flatnonzero(s == 'train')
        val = rng.choice(train_idx, size=max(1, int(len(train_idx) * 0.1)), replace=False)
        s[val] = 'val'
        internal = tgt[['smiles'] + endpoints].copy(); internal['splits'] = s

        rows = [internal[['smiles', 'splits'] + endpoints]] + self._public_rows(data, params, endpoints)
        combined = pd.concat(rows, ignore_index=True)
        for ep in endpoints:                                          # ensure every target column exists
            if ep not in combined:
                combined[ep] = np.nan
        combined = combined[['smiles', 'splits'] + endpoints]

        test_truth = tgt[tgt['compound'].isin(union_test)][['compound', 'smiles'] + endpoints].reset_index(drop=True)
        return combined, test_truth

    def train_predict(self, params, gname, endpoints, combined_path, test_truth, persist=True):
        """chemprop train (HPO hyperparams) on the grouping, then predict the held-out test compounds.
        persist=True copies the trained model dir to output/chemprop_models/<gname>/ (deploy artifact);
        CV folds pass persist=False (throwaway per-fold models)."""
        run = CACHE / 'chemprop_run'; run.mkdir(exist_ok=True)
        model_dir = run / f'sysgrp_model_{gname}'
        test_in = run / f'sysgrp_test_{gname}.csv'
        test_truth[['smiles']].to_csv(test_in, index=False)           # predict input: SMILES only (SMILES-safe file)
        cmd = [cp.CHEMPROP, 'train', '-i', str(combined_path), '-s', 'smiles',
               '--target-columns', *endpoints, '--splits-column', 'splits', '-t', 'regression',
               '--metrics', 'rmse', 'mae', '--epochs', str(params.epochs), '--patience', str(params.patience),
               '--num-workers', '0', '-o', str(model_dir), '--data-seed', str(params.seed)]
        hp = params.hp
        for flag, key in [('--depth', 'depth'), ('--message-hidden-dim', 'message_hidden_dim'),
                          ('--ffn-num-layers', 'ffn_num_layers'), ('--ffn-hidden-dim', 'ffn_hidden_dim'),
                          ('--dropout', 'dropout'), ('-b', 'batch_size'), ('--aggregation', 'aggregation')]:
            if key in hp:
                cmd += [flag, str(hp[key])]
        print(f'  [{gname}] training chemprop ({len(endpoints)} tasks, {params.epochs}ep, hp={hp or "defaults"})...', flush=True)
        cp._run_quiet(cmd, run / f'sysgrp_train_{gname}.log')
        ckpts = list(model_dir.rglob('best*.ckpt')) or list(model_dir.rglob('*.ckpt'))
        ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
        preds_path = run / f'sysgrp_preds_{gname}.csv'
        cp._run_quiet([cp.CHEMPROP, 'predict', '-i', str(test_in), '-s', 'smiles',
                       '--model-path', str(ckpt), '--preds-path', str(preds_path)], run / f'sysgrp_predict_{gname}.log')
        # persist the full trained model dir (for later prediction/deployment) to output/chemprop_models/<gname>/
        if persist:
            saved = MODELS_DIR / gname; shutil.rmtree(saved, ignore_errors=True); saved.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_dir, saved, dirs_exist_ok=True)
        else:
            saved = model_dir
        return pd.read_csv(preds_path), saved

    def eval_grouping(self, data, params, gname, endpoints, preds, test_truth):
        """Score each member endpoint on ITS local temporal test (== RF); save pred_df parquets; return rows."""
        out_dir = ROOT / params.output_dir; rows = []
        for i, ep in enumerate(endpoints):
            col = f'pred_{i}' if f'pred_{i}' in preds.columns else (ep if ep in preds.columns else None)
            yp = preds[col].to_numpy(float) if col else np.full(len(test_truth), np.nan)
            df = pd.DataFrame({'compound': test_truth['compound'], 'real_y': test_truth[ep].to_numpy(float), 'pred_y': yp})
            local = data.temporal_test_ids(ep)                        # ep's OWN newest-30% (RF-matched)
            df = df[df['compound'].isin(local)].dropna(subset=['real_y', 'pred_y']).reset_index(drop=True)
            epdir = out_dir / ep; epdir.mkdir(parents=True, exist_ok=True)
            if len(df):
                df.to_parquet(epdir / f'pred_{gname}_temporal.parquet', index=False)
            r2 = round(float(_rsq(df['real_y'], df['pred_y'])), 3) if len(df) >= 3 and df['real_y'].nunique() > 1 else None
            rho = round(float(spearmanr(df['real_y'], df['pred_y']).statistic), 3) if len(df) >= 3 and df['real_y'].nunique() > 1 else None
            rows.append({'endpoint': ep, 'grouping': gname, 'r2': r2, 'rho': rho, 'n': len(df)})
            print(f'  {ep:11} [{gname:9}] temporal  r2={r2} rho={rho} n={len(df)}', flush=True)
        return rows

    def run_grouping(self, data, params, gname, endpoints):
        combined, test_truth = self.build_combined(data, params, endpoints, np.random.default_rng(params.seed))
        run = CACHE / 'chemprop_run'; run.mkdir(exist_ok=True)
        cpath = run / f'sysgrp_combined_{gname}.csv'; combined.to_csv(cpath, index=False)
        print(f'  [{gname}] combined {len(combined)} rows | internal test compounds={len(test_truth)}', flush=True)
        preds, saved = self.train_predict(params, gname, endpoints, cpath, test_truth)
        rows = self.eval_grouping(data, params, gname, endpoints, preds, test_truth)
        record = {'grouping': gname, 'endpoints': endpoints, 'model_dir': str(saved),
                  'sources': {ep: data.augmented_sources(ep, params) for ep in endpoints},
                  'metrics_r2': {r['endpoint']: r['r2'] for r in rows}, 'n_train': int((combined['splits'] == 'train').sum())}
        return rows, record

    def evaluate_all(self, data, params, groupings=None):
        rows, records = [], []
        for gname in (groupings or list(params.groupings)):
            r, rec = self.run_grouping(data, params, gname, params.groupings[gname])
            rows += r; records.append(rec)
        return rows, records

    # ---- K-fold CV over internal compounds (multitask analog of the RF eval_cv; public always in train) ----
    def run_grouping_cv(self, data, params, gname, endpoints):
        """5-fold CV for one grouping: shared compound-level folds, public always in train, pooled
        out-of-fold predictions per endpoint -> pred_{gname}_cv.parquet + CV r2 (== RF internal_cv_r2)."""
        rng = np.random.default_rng(params.seed)
        ids = data.cv_compound_ids(endpoints)
        kf = KFold(n_splits=params.cv_folds, shuffle=True, random_state=params.seed)
        run = CACHE / 'chemprop_run'; run.mkdir(exist_ok=True)
        oof = {ep: [] for ep in endpoints}
        for k, (_, te) in enumerate(kf.split(ids)):
            combined, test_truth = self.build_combined(data, params, endpoints, rng, test_ids=set(ids[te]))
            cpath = run / f'sysgrpcv_combined_{gname}_f{k}.csv'; combined.to_csv(cpath, index=False)
            print(f'  [{gname} cv fold {k + 1}/{params.cv_folds}] combined {len(combined)} rows | '
                  f'held-out compounds={len(test_truth)}', flush=True)
            preds, _ = self.train_predict(params, f'{gname}_cvf{k}', endpoints, cpath, test_truth, persist=False)
            for i, ep in enumerate(endpoints):
                col = f'pred_{i}' if f'pred_{i}' in preds.columns else (ep if ep in preds.columns else None)
                yp = preds[col].to_numpy(float) if col else np.full(len(test_truth), np.nan)
                oof[ep].append(pd.DataFrame({'compound': test_truth['compound'],
                                             'real_y': test_truth[ep].to_numpy(float), 'pred_y': yp}))
        out_dir = ROOT / params.output_dir; rows = []
        for ep in endpoints:
            df = pd.concat(oof[ep], ignore_index=True).dropna(subset=['real_y', 'pred_y']).reset_index(drop=True)
            epdir = out_dir / ep; epdir.mkdir(parents=True, exist_ok=True)
            if len(df):
                df.to_parquet(epdir / f'pred_{gname}_cv.parquet', index=False)
            ok = len(df) >= 3 and df['real_y'].nunique() > 1
            r2 = round(float(_rsq(df['real_y'], df['pred_y'])), 3) if ok else None
            rho = round(float(spearmanr(df['real_y'], df['pred_y']).statistic), 3) if ok else None
            rows.append({'endpoint': ep, 'grouping': gname, 'cv_r2': r2, 'cv_rho': rho, 'cv_n': len(df)})
            print(f'  {ep:11} [{gname:9}] CV       r2={r2} rho={rho} n={len(df)}', flush=True)
        return rows

    def evaluate_all_cv(self, data, params, groupings):
        rows = []
        for gname in groupings:
            rows += self.run_grouping_cv(data, params, gname, params.groupings[gname])
        return rows

    def write_outputs_cv(self, data, params, rows):
        """summary_chemprop_cv.csv (+ wide). Merges with a prior CV summary so partial grouping runs accrue."""
        out_dir = ROOT / params.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame(rows); prev = out_dir / 'summary_chemprop_cv.csv'
        if prev.exists():
            old = pd.read_csv(prev)
            key = long[['endpoint', 'grouping']].apply(tuple, axis=1)
            old = old[~old[['endpoint', 'grouping']].apply(tuple, axis=1).isin(set(key))]
            long = pd.concat([old, long], ignore_index=True)
        long.to_csv(prev, index=False)
        wide = long.pivot(index='endpoint', columns='grouping', values='cv_r2').reindex(ENDPOINTS)
        wide.to_csv(out_dir / 'summary_chemprop_cv_wide.csv')
        print(f'\n> wrote {prev} (+ wide)', flush=True)
        print('\n=== CV r2 by endpoint x grouping (in-house Pearson, out-of-fold; public always in train) ===', flush=True)
        print(wide.to_string(), flush=True)

    # ---- PUBLIC-ONLY -> INTERNAL (domain-transfer baseline: zero internal compounds in training) ----
    def build_public_only(self, data, params, endpoints, rng):
        """Train on PUBLIC data only: ALL internal compounds -> test (never trained on), public -> train
        (+10% of public carved to val for early stopping). Returns (combined_df, test_truth=all internal)."""
        tgt = data.tgt
        internal = tgt[['smiles'] + endpoints].copy(); internal['splits'] = 'test'   # every internal row held out
        combined = pd.concat([internal[['smiles', 'splits'] + endpoints]] + self._public_rows(data, params, endpoints),
                             ignore_index=True)
        for ep in endpoints:
            if ep not in combined:
                combined[ep] = np.nan
        combined = combined[['smiles', 'splits'] + endpoints]
        train_pos = np.flatnonzero(combined['splits'].to_numpy() == 'train')          # val from PUBLIC (no internal in train)
        val = rng.choice(train_pos, size=max(1, int(len(train_pos) * 0.1)), replace=False)
        combined.iloc[val, combined.columns.get_loc('splits')] = 'val'
        test_truth = tgt[['compound', 'smiles'] + endpoints].reset_index(drop=True)
        return combined, test_truth

    def run_grouping_public_only(self, data, params, gname, endpoints):
        """Train the grouping on public-only, predict ALL internal; score each endpoint on the full internal
        set (external validation); save pred_<grp>_publiconly.parquet. Returns per-endpoint rows."""
        combined, test_truth = self.build_public_only(data, params, endpoints, np.random.default_rng(params.seed))
        run = CACHE / 'chemprop_run'; run.mkdir(exist_ok=True)
        cpath = run / f'sysgrp_pubonly_{gname}.csv'; combined.to_csv(cpath, index=False)
        print(f'  [{gname} public-only] combined {len(combined)} rows (public train) | internal test={len(test_truth)}', flush=True)
        preds, _ = self.train_predict(params, f'{gname}_pubonly', endpoints, cpath, test_truth, persist=False)
        out_dir = ROOT / params.output_dir; rows = []
        for i, ep in enumerate(endpoints):
            col = f'pred_{i}' if f'pred_{i}' in preds.columns else (ep if ep in preds.columns else None)
            yp = preds[col].to_numpy(float) if col else np.full(len(test_truth), np.nan)
            df = pd.DataFrame({'compound': test_truth['compound'], 'real_y': test_truth[ep].to_numpy(float), 'pred_y': yp}) \
                   .dropna(subset=['real_y', 'pred_y']).reset_index(drop=True)
            epdir = out_dir / ep; epdir.mkdir(parents=True, exist_ok=True)
            if len(df):
                df.to_parquet(epdir / f'pred_{gname}_publiconly.parquet', index=False)
            ok = len(df) >= 3 and df['real_y'].nunique() > 1
            r2 = round(float(_rsq(df['real_y'], df['pred_y'])), 3) if ok else None
            rho = round(float(spearmanr(df['real_y'], df['pred_y']).statistic), 3) if ok else None
            rows.append({'endpoint': ep, 'grouping': gname, 'publiconly_r2': r2, 'publiconly_rho': rho, 'publiconly_n': len(df)})
            print(f'  {ep:11} [{gname:9}] PUBLIC->INT r2={r2} rho={rho} n={len(df)}', flush=True)
        return rows

    def evaluate_all_public_only(self, data, params, groupings):
        rows = []
        for gname in groupings:
            rows += self.run_grouping_public_only(data, params, gname, params.groupings[gname])
        return rows

    def write_outputs_public_only(self, data, params, rows):
        out_dir = ROOT / params.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame(rows); prev = out_dir / 'summary_chemprop_publiconly.csv'
        if prev.exists():
            key = set(long[['endpoint', 'grouping']].apply(tuple, axis=1))
            old = pd.read_csv(prev)
            old = old[~old[['endpoint', 'grouping']].apply(tuple, axis=1).isin(key)]
            long = pd.concat([old, long], ignore_index=True)
        long.to_csv(prev, index=False)
        wide = long.pivot(index='endpoint', columns='grouping', values='publiconly_r2').reindex(ENDPOINTS)
        wide.to_csv(out_dir / 'summary_chemprop_publiconly_wide.csv')
        print(f'\n> wrote {prev} (+ wide)', flush=True)
        print('\n=== PUBLIC-ONLY -> INTERNAL r2 by endpoint x grouping (in-house Pearson, ALL internal) ===', flush=True)
        print(wide.to_string(), flush=True)

    # ---- MLTrail registration (needs mltrail; runs in `ML` or `chemprop` env — both have it) ----
    def modelling_unit(self, ep, params):
        c = params.endpoint_cfg[ep]; t, u = c['transform'], c['unit']
        return {'identity': u, 'log10': f'log10({u})', 'logit_pct': 'logit(fraction_unbound)'}.get(t, f'{t}({u})')

    def deploy_sanity(self, registry, model_id, endpoints):
        """Predict PUBLIC reference SMILES end-to-end (via the chemprop CLI) to confirm the model loads."""
        ref = pd.DataFrame({'compound': ['ethanol', 'benzene', 'aspirin'],
                            'smiles': ['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']})
        out = registry.predict(model_id, ref, smiles_column='smiles', compound_id='compound')
        pred_cols = [c for c in out.columns if c not in ('smiles', 'compound')]
        return int(out[pred_cols].notna().all(axis=1).sum()), len(ref)

    def register(self, data, params, registry, groupings=None):
        """Register each saved grouping model dir into MLTrail as a chemprop multitask model (idempotent:
        resets the latest version if `adme_mt_<grp>` already exists), then run a public-SMILES sanity predict.
        `groupings` (if given) restricts to that subset of grouping names."""
        recs = json.loads((ROOT / params.output_dir / 'models_manifest.json').read_text())
        if groupings is not None:
            recs = [r for r in recs if r['grouping'] in set(groupings)]
        listing = registry.list(); out = []
        for rec in recs:
            gname, eps = rec['grouping'], rec['endpoints']
            name = f'adme_mt_{gname}'
            units = {self.modelling_unit(ep, params) for ep in eps}
            hit = listing.loc[listing['experiment_name'] == name, 'id'] if len(listing) else pd.Series([], dtype=int)
            mid = registry.add(
                model_id=int(hit.iloc[0]) if len(hit) else None, overwrite=len(hit) > 0,
                model=rec['model_dir'], experiment_name=name, experiment_measure=','.join(eps),
                unit=units.pop() if len(units) == 1 else 'mixed',
                model_type='multitask_regression' if len(eps) > 1 else 'single_task_regression',
                framework='chemprop', features_type='smiles', target_columns=eps,
                metrics=rec.get('metrics_r2'),
                comment=f'chemprop D-MPNN multitask [{", ".join(eps)}]; augmented+temporal; '
                        f'hp={params.hp or "defaults"}; sources={rec["sources"]}')
            n_ok, n = self.deploy_sanity(registry, mid, eps)
            print(f'  registered {name} (id={mid}) | targets={eps} | sanity {n_ok}/{n} public SMILES predicted', flush=True)
            out.append({'grouping': gname, 'model_id': mid, 'name': name, 'targets': eps})
        return out

    def write_outputs(self, data, params, rows, records):
        out_dir = ROOT / params.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame(rows)
        long.to_csv(out_dir / 'summary_chemprop.csv', index=False)
        wide = long.pivot(index='endpoint', columns='grouping', values='r2').reindex(ENDPOINTS)
        wide.to_csv(out_dir / 'summary_chemprop_wide.csv')
        try:
            commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD']).decode().strip()
        except Exception:
            commit = None
        (out_dir / 'manifest_chemprop.json').write_text(json.dumps(
            {'timestamp': datetime.now().isoformat(timespec='seconds'), 'git_commit': commit,
             'groupings': params.groupings, 'epochs': params.epochs, 'hyperparams': params.hp,
             'temporal_split': 'local_per_endpoint_70_30 (union held-out per grouping)',
             'augmented_sources': {ep: data.augmented_sources(ep, params) for ep in ENDPOINTS}}, indent=2))
        # models_manifest = where each grouping's checkpoints are saved (MLTrail registration deferred until
        # chemprop prediction is supported in MLTrail v1 — see module docstring).
        (out_dir / 'models_manifest.json').write_text(json.dumps(records, indent=2))
        print(f'\n> wrote {out_dir}/summary_chemprop.csv (+ wide) + manifest_chemprop.json', flush=True)
        print(f'> saved {len(records)} chemprop model dirs under {MODELS_DIR} (MLTrail registration deferred)', flush=True)
        print('\n=== r2 by endpoint x grouping (temporal, in-house Pearson) ===', flush=True)
        print(wide.to_string(), flush=True)


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv', action='store_true', help='K-fold CV over internal compounds (public always in train), instead of the temporal split')
    ap.add_argument('--public-only', dest='public_only', action='store_true', help='train each grouping on PUBLIC data only and predict ALL internal (external-validation / domain-transfer baseline)')
    ap.add_argument('--register', action='store_true', help='register saved grouping models into MLTrail (chemprop framework) and sanity-predict')
    ap.add_argument('--groupings', nargs='*', default=None, help='subset of grouping names to run (default: all in config)')
    args = ap.parse_args()
    params = PARAMS(CONFIG)
    data = DATA().load_all(params)
    output = OUTPUT()
    groupings = args.groupings or list(params.groupings)
    if args.register:
        from mltrail import Registry
        reg = output.register(data, params, Registry.from_default(), groupings=args.groupings)
        print(f'> registered {len(reg)} chemprop grouping model(s) into MLTrail', flush=True)
        return
    if args.public_only:
        print(f'> Chemprop PUBLIC-ONLY -> INTERNAL: groupings={groupings} | epochs={params.epochs} | '
              f'hp={params.hp or "defaults"} | train=public only, test=ALL internal', flush=True)
        output.write_outputs_public_only(data, params, output.evaluate_all_public_only(data, params, groupings))
    elif args.cv:
        print(f'> Chemprop CV: groupings={groupings} | folds={params.cv_folds} | epochs={params.epochs} | '
              f'hp={params.hp or "defaults"} | pooled out-of-fold (public always in train)', flush=True)
        output.write_outputs_cv(data, params, output.evaluate_all_cv(data, params, groupings))
    else:
        print(f'> Chemprop systematic: groupings={groupings} | epochs={params.epochs} | '
              f'hp={params.hp or "defaults"} | LOCAL per-endpoint temporal test', flush=True)
        rows, records = output.evaluate_all(data, params, groupings)
        output.write_outputs(data, params, rows, records)


if __name__ == '__main__':
    main()
