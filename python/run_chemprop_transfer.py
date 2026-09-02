#!/usr/bin/env python3
"""Chemprop TRANSFER LEARNING arm: public pretrain -> frozen-encoder multitask finetune on internal.

Attacks the calibration failure of plain augmentation head-on: learn structure->property from the large
public corpus, then RE-CALIBRATE on internal data only.

  Stage 1 (once per grouping, the only expensive step)
      multitask `chemprop train` on PUBLIC ONLY (target-columns = the grouping's endpoints). It never sees
      an internal label, so it cannot leak into any internal fold and is reused by every endpoint/fold.

  Stage 2 (per endpoint, per fold)
      `chemprop train --checkpoint <stage1> --freeze-encoder` on that endpoint's RF fold-TRAIN compounds,
      keeping the SAME multitask target columns (the head shape must match the checkpoint). The finetune
      therefore uses ALL of the train compounds' relevant internal labels (NaN-masked where unmeasured),
      which mimics production. Predict the held-out fold and read only the target endpoint's column.

HONEST EVALUATION: folds are RF's EXACT folds, recovered from the `fold` column of the saved RF
`internal_cv_preddf` (output/results/<date>_metrics/<ep>.pkl). Same compounds, same partition as
RF `internal_cv` -> the only variable is the model.

CAUTION: RF folds differ per endpoint (hlm/mlm fold 324 compounds, rlm only 39), so a compound in hlm's
TRAIN fold can sit in rlm's TEST fold. Hence ONE finetune per (endpoint, fold); auxiliary labels are used
only for compounds in THAT endpoint's fold-train set. Never score several endpoints from one finetune.

Reports R2 (squared Pearson) AND **R2det** (the selection metric) via ML_Reg.get_reg_metrics_from_preddf.
Uncertainty is NOT taken from chemprop — the RF tree-variance conf_* stack is used instead.

Run from env `ML` (shells out to the configured chemprop binary):
  screen -S cptransfer
  ~/miniconda3/envs/ML/bin/python python/run_chemprop_transfer.py --config config/config.yaml \
      --smoke rlm            # single (endpoint, fold) smoke test first
  ~/miniconda3/envs/ML/bin/python python/run_chemprop_transfer.py --config config/config.yaml --resume
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.expanduser('~/Scripts'))

import numpy as np
import pandas as pd
import yaml

import ML_Reg

CACHE = ROOT / 'autoresearch/predict_adme'
PUBLIC_SRC = {'EXP': 'public_{ep}.parquet', 'NVS': 'public_novartis_{ep}.parquet',
              'ADM': 'public_admetlab_{ep}.parquet'}


# ============================================================ PARAMS
class PARAMS:
    """Reads CHEMPROP_TRANSFER from config; the public-source policy is shared with RF_SINGLETASK."""

    def __init__(self, config_path=ROOT / 'config/config.yaml'):
        self.load_params(config_path)

    def load_params(self, config_path):
        cfg = yaml.safe_load(Path(config_path).read_text())
        for k, v in cfg['CHEMPROP_TRANSFER'].items():
            setattr(self, k, v)
        self.chemprop_bin = os.path.expanduser(self.chemprop_bin)
        self.augmented_sources_override = cfg['RF_SINGLETASK'].get('augmented_sources') or {}
        self.seed = cfg['RF_SINGLETASK']['seed']
        return self


# ============================================================ DATA
class DATA:
    """Internal wide targets, RF's exact folds, and the public pretrain rows per grouping."""

    def load_internal(self, params):
        """Internal targets (compound, smiles, 8 endpoint labels) + the 200 precomputed DS_ descriptors."""
        self.tgt = pd.read_parquet(CACHE / 'internal_targets.parquet').reset_index(drop=True)
        self.ds_cols = []
        # precomputed descriptastorus block, joined onto the internal compounds by compound id
        if params.descriptors == 'precomputed':
            import pyarrow.parquet as pq
            fp = ROOT / 'output/features/20260824_MF_features.parquet'
            self.ds_cols = [c for c in pq.read_schema(fp).names if c.startswith('DS_')]
            ds = pd.read_parquet(fp, columns=['compound'] + self.ds_cols)
            self.tgt = self.tgt.merge(ds, on='compound', how='left')
        print(f"> internal targets: {len(self.tgt)} compounds | DS_ descriptors: {len(self.ds_cols)}", flush=True)
        return self

    def augmented_sources(self, ep, params):
        """Config-vetted public sources for ep (same policy as RF): override else every available parquet."""
        srcs = params.augmented_sources_override.get(ep, list(PUBLIC_SRC))
        return [s for s in srcs if (CACHE / PUBLIC_SRC[s].format(ep=ep)).exists()]

    def rf_folds(self, params, ep):
        """RF's EXACT folds for ep, recovered from the `fold` column of its saved internal_cv preddf.
        Returns [[train_ids, test_ids], ...] — identical compounds and partition to the RF internal_cv arm."""
        fp = ROOT / params.rf_metrics_dir / f'{ep}.pkl'
        with open(fp, 'rb') as f:
            loaded = pickle.load(f)
        r = loaded[ep] if isinstance(loaded, dict) and ep in loaded else loaded
        p = r['internal_cv_preddf']
        assert 'fold' in p.columns, f'{ep}: internal_cv_preddf has no fold column'
        ids = p['compound'].to_numpy()
        return [[list(ids[p['fold'].to_numpy() != k]), list(ids[p['fold'].to_numpy() == k])]
                for k in sorted(p['fold'].unique())]

    def public_rows(self, params, endpoints):
        """Public pretrain rows for a grouping: EXP per-endpoint single-target rows, NVS/ADM merged wide.
        Carries the DS_ descriptors when precomputed. Public NEVER enters a test split."""
        keep = (['smiles'] + self.ds_cols) if self.ds_cols else ['smiles']
        rows = []
        # experimental public: one single-target frame per endpoint that allows EXP
        for ep in [e for e in endpoints if 'EXP' in self.augmented_sources(e, params)]:
            r = pd.read_parquet(CACHE / PUBLIC_SRC['EXP'].format(ep=ep), columns=keep + ['value'])
            rows.append(r.rename(columns={'value': ep}))
        # predicted public (Novartis / ADMETlab): merge the grouping's endpoints wide on smiles
        for key in ('NVS', 'ADM'):
            eps = [e for e in endpoints if key in self.augmented_sources(e, params)]
            w = None
            for ep in eps:
                r = pd.read_parquet(CACHE / PUBLIC_SRC[key].format(ep=ep), columns=keep + ['value'])
                r = r.rename(columns={'value': ep}).drop_duplicates('smiles')
                w = r if w is None else w.merge(r[['smiles', ep]], on='smiles', how='outer')
            if w is not None:
                rows.append(w)
        return rows


# ============================================================ OUTPUT
class OUTPUT:
    """Runs the two chemprop stages and scores each endpoint on RF's folds."""

    def __init__(self, params):
        self.run = CACHE / 'chemprop_transfer_run'
        self.run.mkdir(exist_ok=True)
        self.rows = []

    # ---------- chemprop plumbing ----------

    def _hp_flags(self, params):
        """Map the config hp dict onto chemprop train flags."""
        return [x for flag, key in [('--depth', 'depth'), ('--message-hidden-dim', 'message_hidden_dim'),
                                    ('--ffn-num-layers', 'ffn_num_layers'), ('--ffn-hidden-dim', 'ffn_hidden_dim'),
                                    ('--dropout', 'dropout'), ('-b', 'batch_size'), ('--aggregation', 'aggregation')]
                if key in params.hp for x in (flag, str(params.hp[key]))]

    def _desc_flags(self, params, data, csv_cols):
        """Descriptor flags — MUST be identical at pretrain, finetune and predict (v2 stores only the scaler)."""
        if params.descriptors == 'precomputed' and data.ds_cols:
            present = [c for c in data.ds_cols if c in csv_cols]
            return ['--descriptors-columns', *present] if present else []
        if params.descriptors == 'featurizer':
            return ['--molecule-featurizers', *params.molecule_featurizers]
        return []

    def _run(self, cmd, log):
        """Run a chemprop command, tee-ing its output to a log; raise with the log tail on failure."""
        with open(log, 'w') as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            tail = ''.join(open(log).readlines()[-25:])
            raise RuntimeError(f'chemprop failed ({p.returncode}); log {log}\n{tail}')

    @staticmethod
    def _ckpt(model_dir):
        """Newest checkpoint written by chemprop train."""
        c = list(Path(model_dir).rglob('best*.ckpt')) or list(Path(model_dir).rglob('*.ckpt'))
        if not c:
            raise FileNotFoundError(f'no checkpoint under {model_dir}')
        return max(c, key=lambda p: p.stat().st_mtime)

    # ---------- stage 1: public-only multitask pretrain ----------

    def pretrain(self, data, params, gname, endpoints, force=False, n_rows=None, tag=''):
        """Multitask chemprop on PUBLIC ONLY -> a reusable checkpoint. Never sees an internal label.
        n_rows subsamples the public corpus (smoke runs only); tag isolates the smoke checkpoint dir."""
        out = ROOT / params.pretrain_dir / (gname + tag)
        if out.exists() and not force:
            try:
                ck = self._ckpt(out)
                print(f"> [{gname}] pretrain checkpoint exists -> {ck.relative_to(ROOT)}", flush=True)
                return ck
            except FileNotFoundError:
                pass
        rows = data.public_rows(params, endpoints)
        pub = pd.concat(rows, ignore_index=True)
        for ep in endpoints:                                    # every target column must exist
            if ep not in pub:
                pub[ep] = np.nan
        pub = pub.dropna(subset=endpoints, how='all').reset_index(drop=True)
        # smoke runs train on a small public subsample so the pipeline can be validated in minutes
        if n_rows and len(pub) > n_rows:
            pub = pub.sample(n=n_rows, random_state=params.seed).reset_index(drop=True)
            print(f"  [{gname}] SMOKE: subsampled public to {len(pub)} rows", flush=True)
        # 10% of the public rows held out as chemprop's validation split (no test split needed here)
        rng = np.random.default_rng(params.seed)
        sp = np.array(['train'] * len(pub), dtype=object)
        sp[rng.choice(len(pub), size=max(1, int(len(pub) * 0.1)), replace=False)] = 'val'
        pub['splits'] = sp
        cols = ['smiles', 'splits'] + endpoints + [c for c in data.ds_cols if c in pub.columns]
        csv = self.run / f'pretrain_{gname}{tag}.csv'
        pub[cols].to_csv(csv, index=False)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [params.chemprop_bin, 'train', '-i', str(csv), '-s', 'smiles',
               '--target-columns', *endpoints, '--splits-column', 'splits', '-t', 'regression',
               '--metrics', 'rmse', 'mae', '--epochs', str(params.epochs_pretrain),
               '--patience', str(params.patience), '--num-workers', '0',
               '-o', str(out), '--data-seed', str(params.seed)]
        cmd += self._hp_flags(params) + self._desc_flags(params, data, set(cols))
        print(f"> [{gname}] PRETRAIN on {len(pub)} public rows, {len(endpoints)} tasks, "
              f"{params.epochs_pretrain}ep ...", flush=True)
        t = time.perf_counter()
        self._run(cmd, self.run / f'pretrain_{gname}{tag}.log')
        ck = self._ckpt(out)
        print(f"  done in {time.perf_counter() - t:.0f}s -> {ck.relative_to(ROOT)}", flush=True)
        return ck

    # ---------- stage 2: frozen-encoder finetune on one RF fold ----------

    def finetune_fold(self, data, params, ep, gname, endpoints, ckpt, fold, train_ids, test_ids, arm='transfer'):
        """Train on ep's RF fold-TRAIN compounds (using ALL their grouping labels) and predict the held-out
        fold. Returns a preddf (compound, real_y, pred_y) for ep only.
        ckpt set   -> `--checkpoint` (+ `--freeze-encoder`): the public-pretrained transfer arm.
        ckpt None  -> the from-scratch CONTROL: internal fold-train only, no public, no pretrained weights,
                      so the whole D-MPNN trains. Isolates what the pretraining actually contributes."""
        tgt = data.tgt
        tr = tgt[tgt['compound'].isin(set(train_ids))]
        te = tgt[tgt['compound'].isin(set(test_ids))]
        # train/val split inside the fold-train compounds only (test rows are the held-out fold)
        rng = np.random.default_rng(params.seed + fold)
        sp = np.array(['train'] * len(tr), dtype=object)
        if len(tr) >= 10:
            sp[rng.choice(len(tr), size=max(1, int(len(tr) * 0.1)), replace=False)] = 'val'
        ft = pd.concat([tr.assign(splits=sp), te.assign(splits='test')], ignore_index=True)
        cols = ['smiles', 'splits'] + endpoints + [c for c in data.ds_cols if c in ft.columns]
        csv = self.run / f'ft_{arm}_{ep}_f{fold}.csv'
        ft[cols].to_csv(csv, index=False)
        mdir = self.run / f'ft_model_{arm}_{ep}_f{fold}'
        cmd = [params.chemprop_bin, 'train', '-i', str(csv), '-s', 'smiles',
               '--target-columns', *endpoints, '--splits-column', 'splits', '-t', 'regression',
               '--metrics', 'rmse', 'mae', '--epochs', str(params.epochs_finetune),
               '--patience', str(params.patience), '--num-workers', '0',
               '-o', str(mdir), '--data-seed', str(params.seed)]
        # transfer arm loads the public-pretrained weights; the control trains from scratch
        if ckpt is not None:
            cmd += ['--checkpoint', str(ckpt)]
            if params.freeze_encoder:
                cmd += ['--freeze-encoder']
        cmd += self._hp_flags(params) + self._desc_flags(params, data, set(cols))
        self._run(cmd, self.run / f'ft_{arm}_{ep}_f{fold}.log')
        # predict the held-out fold with the finetuned model
        pin = self.run / f'ft_test_{arm}_{ep}_f{fold}.csv'
        te[['smiles'] + [c for c in data.ds_cols if c in te.columns]].to_csv(pin, index=False)
        pout = self.run / f'ft_preds_{arm}_{ep}_f{fold}.csv'
        pcmd = [params.chemprop_bin, 'predict', '-i', str(pin), '-s', 'smiles',
                '--model-path', str(self._ckpt(mdir)), '--preds-path', str(pout)]
        pcmd += self._desc_flags(params, data, set(te.columns))
        self._run(pcmd, self.run / f'ftpred_{arm}_{ep}_f{fold}.log')
        preds = pd.read_csv(pout)
        # the endpoint's column: multitask predictions come back as pred_<i> or named by target
        i = endpoints.index(ep)
        col = f'pred_{i}' if f'pred_{i}' in preds.columns else (ep if ep in preds.columns else None)
        if col is None:
            raise RuntimeError(f'{ep} f{fold}: no prediction column in {list(preds.columns)[:6]}')
        return pd.DataFrame({'compound': te['compound'].to_numpy(), 'real_y': te[ep].to_numpy(float),
                             'pred_y': preds[col].to_numpy(float), 'fold': fold})

    # ---------- per-endpoint driver ----------

    def run_endpoint(self, data, params, ep, ckpt, gname, endpoints, folds, max_folds=None, arm='transfer'):
        """Loop ep's RF folds, pool the out-of-fold predictions, and score with R2 + R2det.
        arm 'transfer' uses the pretrained checkpoint; arm 'scratch' is the internal-only control."""
        parts = []
        for k, (tr, te) in enumerate(folds, 1):
            if max_folds and k > max_folds:
                break
            t = time.perf_counter()
            pdf = self.finetune_fold(data, params, ep, gname, endpoints, ckpt, k, tr, te, arm=arm)
            parts.append(pdf)
            print(f"  [{arm}/{ep} fold {k}/{len(folds)}] n_train={len(tr)} n_test={len(te)} "
                  f"({time.perf_counter() - t:.0f}s)", flush=True)
        oof = pd.concat(parts, ignore_index=True).dropna(subset=['real_y', 'pred_y']).reset_index(drop=True)
        m = ML_Reg.get_reg_metrics_from_preddf(oof, ntrain=len(data.tgt))
        out = ROOT / params.output_dir
        (out / 'preddfs').mkdir(parents=True, exist_ok=True)
        oof.to_parquet(out / 'preddfs' / f'{ep}_{arm}_cv.parquet', index=False)
        rec = {'endpoint': ep, 'arm': arm, 'grouping': gname if arm == 'transfer' else '-', 'n': len(oof),
               'r2': m.get('r2'), 'r2det': m.get('r2det'), 'rmse': m.get('rmse'),
               'spearman_rho': m.get('spearman_rho'),
               'freeze_encoder': bool(params.freeze_encoder) if arm == 'transfer' else False,
               'folds_run': len(parts)}
        print(f"> {ep:11} [{arm:8}] r2={rec['r2']}  R2det={rec['r2det']}  n={rec['n']}", flush=True)
        return rec

    def write_outputs(self, params):
        """Stream the cross-endpoint summary (merges with a prior run so partial runs accrue)."""
        out = ROOT / params.output_dir
        out.mkdir(parents=True, exist_ok=True)
        fp = out / 'summary_transfer.csv'
        tbl = pd.DataFrame(self.rows)
        if fp.exists():
            prev = pd.read_csv(fp)
            keys = set(zip(tbl['endpoint'], tbl['arm']))
            if 'arm' in prev.columns:
                prev = prev[[k not in keys for k in zip(prev['endpoint'], prev['arm'])]]
            tbl = pd.concat([prev, tbl], ignore_index=True)
        tbl.to_csv(fp, index=False)
        return tbl, fp


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser(description="Chemprop transfer arm: public pretrain -> frozen finetune on RF folds.")
    ap.add_argument('--config', default=str(ROOT / 'config/config.yaml'))
    ap.add_argument('--endpoints', default='all', help="comma list or 'all'")
    ap.add_argument('--smoke', default='', help="single endpoint, 1 fold only (validate the pipeline cheaply)")
    ap.add_argument('--force_pretrain', action='store_true', help="retrain stage 1 even if a checkpoint exists")
    ap.add_argument('--smoke_rows', type=int, default=2000, help="public rows for a --smoke pretrain")
    ap.add_argument('--smoke_epochs', type=int, default=3, help="epochs for both stages during --smoke")
    ap.add_argument('--arms', default='transfer,scratch',
                    help="transfer = public pretrain + frozen finetune; scratch = internal-only control (no pretrain)")
    ap.add_argument('--resume', action='store_true', help="skip endpoints already in summary_transfer.csv")
    args = ap.parse_args()

    params = PARAMS(args.config).load_params(args.config)
    data = DATA().load_internal(params)
    output = OUTPUT(params)

    eps = [args.smoke] if args.smoke else (list(params.endpoint_grouping) if args.endpoints == 'all'
                                           else args.endpoints.split(','))
    arms = [a for a in args.arms.split(',') if a]
    done = set()
    fp = ROOT / params.output_dir / 'summary_transfer.csv'
    if args.resume and fp.exists():
        prev = pd.read_csv(fp)
        done = set(zip(prev['endpoint'], prev['arm'])) if 'arm' in prev.columns else set()

    # a smoke run shrinks BOTH stages so the pipeline is validated in minutes, not hours
    if args.smoke:
        params.epochs_pretrain = params.epochs_finetune = args.smoke_epochs
    print(f"> chemprop transfer | endpoints={eps} | arms={arms} | freeze_encoder={params.freeze_encoder} | "
          f"descriptors={params.descriptors} | hp={params.hp}"
          + (f" | SMOKE {args.smoke_rows} rows / {args.smoke_epochs} ep / 1 fold" if args.smoke else ''), flush=True)

    # stage 1 once per grouping — skipped entirely when only the scratch control is requested
    ckpts = {}
    if 'transfer' in arms:
        need = sorted({params.endpoint_grouping[e] for e in eps})
        ckpts = {g: output.pretrain(data, params, g, params.groupings[g], force=args.force_pretrain,
                                    n_rows=args.smoke_rows if args.smoke else None,
                                    tag='_smoke' if args.smoke else '') for g in need}

    # stage 2 per endpoint on RF's exact folds
    for ep in eps:
        g = params.endpoint_grouping[ep]
        folds = data.rf_folds(params, ep)
        for arm in arms:
            if (ep, arm) in done:
                print(f"[skip] {ep}/{arm}: already in summary", flush=True); continue
            rec = output.run_endpoint(data, params, ep, ckpts.get(g) if arm == 'transfer' else None,
                                      g, params.groupings[g], folds,
                                      max_folds=1 if args.smoke else None, arm=arm)
            output.rows.append(rec)
            if not args.smoke:
                output.write_outputs(params)

    if args.smoke:
        print("\n===== SMOKE RESULT (1 fold, tiny pretrain — validity check only, NOT a score) =====")
        print(pd.DataFrame(output.rows).to_string(index=False))
        print("\npipeline OK: checkpoint -> freeze-encoder finetune -> predict -> R2det. Not written to summary.")
        return
    tbl, fp = output.write_outputs(params)
    print("\n===== CHEMPROP on RF's EXACT folds =====")
    print(tbl.to_string(index=False))
    # transfer-vs-control on the selection metric (the question the control exists to answer)
    if {'transfer', 'scratch'} <= set(tbl['arm']):
        w = tbl.pivot_table(index='endpoint', columns='arm', values='r2det')
        if {'transfer', 'scratch'} <= set(w.columns):
            w['pretrain_gain'] = (w['transfer'] - w['scratch']).round(4)
            print("\nR2det — does public pretraining help?")
            print(w.to_string())
    print(f"\n-> {fp}")


if __name__ == '__main__':
    main()
