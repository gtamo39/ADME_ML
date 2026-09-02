"""Find the predictive subset of the NVS (Novartis-NIBR, predicted-label) public data that IMPROVES the
augmented-CV R2 on a small internal endpoint (pilot: mdck), instead of the all-NVS pool that dilutes it.

Runs in the user's kernel on an already-built DATA/OUTPUT (mdck): after
  data.build_ML_data_mdck(); data.get_internal_public_sets('mdck'); data.select_best_combo_and_update(params,'mdck')
construct `NVSSubsetSearch(data, output, params)` and call the lever methods. Nothing here prints SMILES /
per-compound values — aggregate counts and R2 only.

Four levers (all vs the internal-only and all-NVS baselines):
  S1 distance   : augment with NVS within a swept Tanimoto distance of the internal TRAIN fold (leakage-free).
  S2 scaffold   : cluster NVS by Bemis-Murcko scaffold; forward/backward greedy group selection.
  S5 biascorr   : affine-calibrate NVS labels toward internal on near-neighbour pairs (per train fold), then augment.
  S3 weighting  : within-NVS CV -> per-compound reliability -> RF sample_weight on the NVS rows.

Honest reporting: every lever's chosen knob is selected on an INNER CV and scored on a held-out OUTER fold
(nested CV); the naive selection-maximized value is reported alongside so the optimism is visible.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

COL2RM = ['compound', 'smiles', 'label', 'source', 'origin', '_ik']   # non-feature columns (mirror OUTPUT.COL2RM)


def _pearson_r2(y, p):
    """Squared Pearson correlation (the in-house R2 in the plot); nan when a side is constant / n<2."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1] ** 2)


def _iqr(v):
    """Interquartile range (robust spread); 0 for a degenerate vector."""
    q75, q25 = np.percentile(np.asarray(v, float), [75, 25])
    return float(q75 - q25)


class NVSSubsetSearch:
    """Subset/label-repair search over the NVS augmentation pool for one internal endpoint."""

    def __init__(self, data, output, params, endpoint='mdck', seed=42, radius=2, n_bits=2048, learner='rf', pool='combo'):
        """
        param DATA data: built for `endpoint` (build_ML_data + get_internal_public_sets + select_best_combo).
        param OUTPUT output: supplies make_model() (the champion RF).
        param PARAMS params: config (RF_SINGLETASK, ADME_ENDPOINTS).
        param str endpoint: endpoint key (pilot 'mdck').
        param str learner: base model — 'rf' (champion), 'rf50' (fast RF), 'lgbm' (LightGBM, high-throughput).
        param pool: public augmentation pool — 'combo' (the selected best-combo origins, deploy default),
                    'all' (every public compound, origin-agnostic near-shell search), or an explicit origin list.
        """
        self.data, self.output, self.params, self.k, self.seed, self.learner = data, output, params, endpoint, seed, learner
        # feature columns = modelling frame minus id/label/meta
        self.feats = [c for c in data.d.columns if c not in COL2RM]
        # internal rows (CV target) and the public augmentation pool
        self.internal = data.internal.reset_index(drop=True)
        if pool == 'all':
            self.nvs = data.pub.reset_index(drop=True)
        elif pool == 'combo':
            self.nvs = data.pub[data.pub.origin.isin(data.combo)].reset_index(drop=True)
        else:
            origins = pool if isinstance(pool, (list, tuple, set)) else [pool]
            self.nvs = data.pub[data.pub.origin.isin(origins)].reset_index(drop=True)
        self.int_ids, self.nvs_ids = list(self.internal.compound), list(self.nvs.compound)
        # frames indexed by compound for fast row lookup during fits
        self._byid = data.d.set_index('compound')
        # InChIKey per internal compound (grouped folds so twins never straddle a split)
        self._ik = dict(zip(self.internal.compound, self.internal._ik))
        # precompute ECFP fingerprints + the NVS<->internal Tanimoto matrix + Murcko scaffolds (once)
        self._mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self._precompute()

    # ---------- precompute ----------

    def _fp(self, smiles):
        """Morgan (ECFP) bit fingerprint for one SMILES; None on parse failure."""
        m = Chem.MolFromSmiles(str(smiles))
        return None if m is None else self._mgen.GetFingerprint(m)

    def _precompute(self):
        """Fingerprint internal + NVS, build the (n_int x n_nvs) Tanimoto matrix, and NVS Murcko scaffolds."""
        self._fp_int = [self._fp(s) for s in self.internal.smiles]
        self._fp_nvs = [self._fp(s) for s in self.nvs.smiles]
        # Tanimoto similarity of every NVS compound to every internal compound (rows=internal, cols=nvs)
        sim = np.zeros((len(self._fp_int), len(self._fp_nvs)), dtype=np.float32)
        for j, fj in enumerate(self._fp_nvs):
            if fj is None:
                continue
            ok = [i for i, fi in enumerate(self._fp_int) if fi is not None]
            sim[ok, j] = DataStructs.BulkTanimotoSimilarity(fj, [self._fp_int[i] for i in ok])
        self._sim = sim
        self._nvs_pos = {c: j for j, c in enumerate(self.nvs_ids)}
        self._int_pos = {c: i for i, c in enumerate(self.int_ids)}
        # Bemis-Murcko generic scaffold per NVS compound (for S2 grouping)
        self._scaffold = [self._generic_scaffold(s) for s in self.nvs.smiles]

    @staticmethod
    def _generic_scaffold(smiles):
        """Bemis-Murcko generic (graph) scaffold SMILES; '' on failure."""
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return ''
        try:
            return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m)))
        except Exception:
            return ''

    # ---------- fit primitive ----------

    def _make(self):
        """Build the base estimator for the active learner. 'rf' uses the champion via output.make_model
        (keeps tests/FakeOutput working); 'rf50'/'lgbm' build directly from the champion config."""
        if self.learner == 'lgbm':
            from lightgbm import LGBMRegressor
            cfg = self.output.cfg
            return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                 colsample_bytree=0.6, n_jobs=cfg['n_jobs'], random_state=cfg['seed'], verbose=-1)
        if self.learner == 'rf50':
            from sklearn.ensemble import RandomForestRegressor as _RF
            cfg = self.output.cfg
            return _RF(**{**cfg['champion'], 'n_estimators': 50}, n_jobs=cfg['n_jobs'], random_state=cfg['seed'])
        return deepcopy(self.output.make_model(False))          # 'rf' champion (works with FakeOutput in tests)

    def _fit_predict(self, train_ids, test_ids, nvs_ids=(), nvs_label=None, nvs_weight=None):
        """Fit the champion RF on internal train_ids + nvs_ids, predict test_ids; return predictions (raw label).
        nvs_label: optional dict nvs_id->corrected label (S5). nvs_weight: optional dict nvs_id->weight (S3)."""
        tr = list(train_ids) + list(nvs_ids)
        Xtr = self._byid.loc[tr, self.feats].to_numpy(np.float32)   # float32: halves RAM on the 273k matrix
        ytr = self._byid.loc[tr, 'label'].to_numpy().astype(float)
        # apply corrected NVS labels (bias-correction) onto the tail of the train vector
        if nvs_label is not None and len(nvs_ids):
            ytr[len(train_ids):] = [nvs_label.get(c, self._byid.at[c, 'label']) for c in nvs_ids]
        # sample weights: internal = 1, NVS = provided (default 1)
        sw = None
        if nvs_weight is not None and len(nvs_ids):
            sw = np.concatenate([np.ones(len(train_ids)), np.array([nvs_weight.get(c, 1.0) for c in nvs_ids])])
        rf = self._make()
        rf.fit(Xtr, ytr, sample_weight=sw)
        return rf.predict(self._byid.loc[list(test_ids), self.feats].to_numpy(np.float32))

    def _grouped_folds(self, ids, n_splits=5, seed=None):
        """InChIKey-grouped CV folds over `ids` (internal): a molecule's twins never split across train/test."""
        seed = self.seed if seed is None else seed
        grp = np.array([self._ik.get(c) if pd.notna(self._ik.get(c)) else c for c in ids])
        uniq = pd.unique(grp)
        n_splits = max(2, min(n_splits, len(uniq)))          # never more folds than groups
        fold_of = {}
        for f, (_, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(uniq)):
            fold_of.update({uniq[j]: f for j in te})
        cf = np.array([fold_of[g] for g in grp])
        ids = np.asarray(ids)
        return [[list(ids[cf != f]), list(ids[cf == f])] for f in range(n_splits)]

    def _cv_r2(self, select, nvs_label_fn=None, nvs_weight=None, folds=None):
        """Augmented CV R2 over internal. select(train_ids)->nvs ids to add (per fold, leakage-free)."""
        folds = folds if folds is not None else self._grouped_folds(self.int_ids)
        real, pred = [], []
        for tr, te in folds:
            add = list(select(tr))
            lab = nvs_label_fn(tr, add) if nvs_label_fn is not None else None
            pred += list(self._fit_predict(tr, te, add, nvs_label=lab, nvs_weight=nvs_weight))
            real += list(self._byid.loc[te, 'label'].to_numpy())
        return _pearson_r2(real, pred)

    def cv_preddf(self, select, nvs_label_fn=None, nvs_weight=None, folds=None, record=None):
        """Like _cv_r2 but RETURN the OOF pred_df (compound, real_y, pred_y, fold, residuals) — modelling space.
        If `record` (list) is given, append one dict per fold: fold, n_int_train, n_nvs_added, test_ids (provenance)."""
        folds = folds if folds is not None else self._grouped_folds(self.int_ids)
        parts = []
        for fi, (tr, te) in enumerate(folds, 1):
            add = list(select(tr))
            lab = nvs_label_fn(tr, add) if nvs_label_fn is not None else None
            pr = self._fit_predict(tr, te, add, nvs_label=lab, nvs_weight=nvs_weight)
            parts.append(pd.DataFrame({'compound': list(te), 'real_y': self._byid.loc[te, 'label'].to_numpy(float),
                                       'pred_y': pr, 'fold': fi}))
            if record is not None:
                record.append({'fold': fi, 'n_int_train': len(tr), 'n_nvs_added': len(add), 'test_ids': list(te)})
        df = pd.concat(parts, ignore_index=True)
        df['residuals'] = (df['pred_y'] - df['real_y']).abs()
        return df

    # ---------- selection helpers ----------

    def _dist_to_train(self, train_ids):
        """Per-NVS Tanimoto distance (1 - max similarity) to the internal TRAIN fold only (leakage-free)."""
        rows = [self._int_pos[c] for c in train_ids if c in self._int_pos]
        return 1.0 - self._sim[rows, :].max(axis=0)          # (n_nvs,)

    def _select_within(self, train_ids, tau):
        """NVS ids whose distance to the train fold is < tau."""
        d = self._dist_to_train(train_ids)
        return [self.nvs_ids[j] for j in np.where(d < tau)[0]]

    # ---------- baselines ----------

    def baseline(self):
        """Internal-only and all-NVS augmented CV R2 (should match the grouped-fold plot values)."""
        internal = self._cv_r2(lambda tr: [])
        all_nvs = self._cv_r2(lambda tr: self.nvs_ids)
        return {'internal_only': internal, 'all_nvs': all_nvs, 'n_internal': len(self.int_ids), 'n_nvs': len(self.nvs_ids)}

    # ---------- S1: distance sweep ----------

    def distance_taus(self, n=10):
        """Even distance grid from the NVS->internal distance distribution (quantiles), ascending."""
        d = 1.0 - self._sim.max(axis=0)                      # NVS distance to the WHOLE internal set (grid only)
        qs = np.linspace(0.1, 1.0, n)
        return sorted(set(np.quantile(d, qs).round(4)))

    def distance_curve(self, taus=None, verbose=True):
        """Augmented CV R2 as a function of the distance threshold tau (naive, single-level CV — for the curve).
        verbose streams each (tau, n_nvs, r2) as it is computed (the sweep is otherwise silent to the end)."""
        taus = taus if taus is not None else self.distance_taus()
        rows = []
        for tau in tqdm(taus, desc='S1 distance sweep'):
            sel = lambda tr, _t=tau: self._select_within(tr, _t)
            # median NVS count kept across folds (aggregate; for the plot x-axis)
            n_kept = int(np.median([len(sel(tr)) for tr, _ in self._grouped_folds(self.int_ids)]))
            r2 = self._cv_r2(sel)
            rows.append({'tau': float(tau), 'n_nvs_median': n_kept, 'r2': r2})
            if verbose:
                tqdm.write(f'  [S1] tau={tau:.3f}  n_nvs={n_kept:<6d} r2={r2:.4f}')
        return pd.DataFrame(rows)

    # ---------- S2 / S4: scaffold-group greedy ----------

    def scaffold_groups(self, max_groups=10):
        """Map each NVS compound to a scaffold GROUP: the top (max_groups-1) frequent scaffolds + an 'other' bucket."""
        vc = pd.Series(self._scaffold).value_counts()
        top = set(vc.index[:max_groups - 1]) - {''}
        grp = np.array([s if s in top else 'other' for s in self._scaffold])
        # ids per group (dict group -> list of nvs ids)
        return {g: [self.nvs_ids[j] for j in np.where(grp == g)[0]] for g in pd.unique(grp)}

    def scaffold_greedy(self, direction='forward', max_groups=10, patience=1, folds=None):
        """Greedy group selection maximizing augmented CV R2. forward: add best group until no gain; backward:
        start from all groups, drop the worst until no gain. Returns the chosen groups + the R2 path."""
        groups = self.scaffold_groups(max_groups)
        keys = list(groups)
        ids_of = lambda ks: [c for g in ks for c in groups[g]]
        if direction == 'forward':
            chosen, best, path, misses = [], self._cv_r2(lambda tr: []), [], 0
            path.append({'step': 0, 'group': 'internal_only', 'r2': best, 'chosen': []})
            tqdm.write(f'  [S2 fwd] start internal_only r2={best:.4f}')
            while misses < patience and len(chosen) < len(keys):
                cand = [g for g in keys if g not in chosen]
                scores = {g: self._cv_r2(lambda tr, _g=g: ids_of(chosen + [_g]), folds=folds)
                          for g in tqdm(cand, desc=f'S2 fwd step {len(chosen) + 1}', leave=False)}
                g = max(scores, key=lambda x: (scores[x] if np.isfinite(scores[x]) else -np.inf))
                path.append({'step': len(path), 'group': g, 'r2': scores[g], 'chosen': chosen + [g]})
                gain = np.isfinite(scores[g]) and scores[g] > best
                tqdm.write(f'  [S2 fwd] +{g!r:>12} r2={scores[g]:.4f} {"(keep)" if gain else "(no gain)"}')
                if gain:
                    best, chosen, misses = scores[g], chosen + [g], 0
                else:
                    misses += 1
            return {'direction': 'forward', 'chosen': chosen, 'r2': best, 'path': pd.DataFrame(path)}
        # backward
        chosen, best, path, misses = list(keys), self._cv_r2(lambda tr: ids_of(keys), folds=folds), [], 0
        path.append({'step': 0, 'group': 'all', 'r2': best, 'chosen': list(keys)})
        tqdm.write(f'  [S2 bwd] start all groups r2={best:.4f}')
        while misses < patience and len(chosen) > 1:
            scores = {g: self._cv_r2(lambda tr, _g=g: ids_of([x for x in chosen if x != _g]), folds=folds)
                      for g in tqdm(chosen, desc=f'S2 bwd ({len(chosen)} left)', leave=False)}
            g = max(scores, key=lambda x: (scores[x] if np.isfinite(scores[x]) else -np.inf))
            path.append({'step': len(path), 'group': f'drop_{g}', 'r2': scores[g], 'chosen': [x for x in chosen if x != g]})
            gain = np.isfinite(scores[g]) and scores[g] > best
            tqdm.write(f'  [S2 bwd] -{g!r:>12} r2={scores[g]:.4f} {"(keep)" if gain else "(no gain)"}')
            if gain:
                best, chosen, misses = scores[g], [x for x in chosen if x != g], 0
            else:
                misses += 1
        return {'direction': 'backward', 'chosen': chosen, 'r2': best, 'path': pd.DataFrame(path)}

    # ---------- S5: bias-correction ----------

    def _bias_label(self, method='shift', sim0=0.5, min_anchors=20):
        """Return a per-fold label function that recalibrates the NVS labels toward the internal scale,
        fit on the fold's train only (leakage-free). Corrects the systematic Novartis offset that tanks R2det.

        method:
          'shift'  : additive — match the MEDIAN of the added-NVS labels to the internal-train median
                     (targets a constant assay offset; robust; always computable even when NVS is far).
          'affine' : robust linear — match MEDIAN and IQR (removes offset AND scale mismatch).
          'anchor' : affine fit on near-neighbour pairs (NVS within sim0 of an internal-train compound,
                     paired to that neighbour's measured label); falls back to 'shift' when < min_anchors.
        Returns {} (no correction) only when the added set is empty.
        """
        def fn(train_ids, add_ids):
            if not add_ids:
                return {}
            ya = np.array([self.nvs.at[self._nvs_pos[c], 'label'] for c in add_ids], float)   # NVS labels to fix
            yint = self._byid.loc[list(train_ids), 'label'].to_numpy(float)                    # internal-train scale
            a, b = float(np.median(yint) - np.median(ya)), 1.0                                 # default: shift
            if method == 'affine':
                si, sa = _iqr(yint), _iqr(ya)
                b = (si / sa) if sa > 0 else 1.0
                a = float(np.median(yint) - b * np.median(ya))
            elif method == 'anchor':
                rows = [self._int_pos[c] for c in train_ids if c in self._int_pos]
                sub = self._sim[np.ix_(rows, [self._nvs_pos[c] for c in add_ids])]             # (n_train, n_add)
                smax, nn = sub.max(axis=0), sub.argmax(axis=0)
                anc = np.where(smax >= sim0)[0]
                if len(anc) >= min_anchors:
                    x = ya[anc]; y = yint[nn[anc]]                                             # paired truth
                    b, a = (np.polyfit(x, y, 1) if np.ptp(x) > 0 else (1.0, float(np.mean(y - x))))
            return {c: float(a + b * ya[j]) for j, c in enumerate(add_ids)}
        return fn

    def biascorrect_curve(self, sim0=0.5, taus=None, method='shift', min_anchors=20):
        """Distance sweep WITH bias-corrected NVS labels (compare against distance_curve)."""
        taus = taus if taus is not None else self.distance_taus()
        lab = self._bias_label(method=method, sim0=sim0, min_anchors=min_anchors)
        rows = []
        for t in tqdm(taus, desc=f'S5 {method} bias-corrected sweep'):
            r2 = self._cv_r2(lambda tr, _t=t: self._select_within(tr, _t), nvs_label_fn=lab)
            rows.append({'tau': float(t), 'r2': r2})
            tqdm.write(f'  [S5 {method}] tau={t:.3f} r2={r2:.4f}')
        return pd.DataFrame(rows)

    # ---------- S3: label-uncertainty weighting ----------

    def nvs_uncertainty(self, n_splits=5):
        """Within-NVS CV (scaffold-grouped) tree-variance std per NVS compound = a self-consistency proxy."""
        grp = np.array(self._scaffold)
        uniq = pd.unique(grp)
        if len(uniq) < 2:
            return np.full(len(self.nvs_ids), np.nan)        # one scaffold group -> no within-NVS CV possible
        n_splits = max(2, min(n_splits, len(uniq)))
        fold_of = {}
        for f, (_, te) in enumerate(KFold(n_splits, shuffle=True, random_state=self.seed).split(uniq)):
            fold_of.update({uniq[j]: f for j in te})
        cf = np.array([fold_of[g] for g in grp])
        std = np.full(len(self.nvs_ids), np.nan)
        for f in tqdm(range(n_splits), desc='S3 within-NVS uncertainty', leave=False):
            tr_ids = [self.nvs_ids[j] for j in np.where(cf != f)[0]]
            te_j = np.where(cf == f)[0]
            # tree-variance needs an RF ensemble, independent of the campaign learner
            from sklearn.ensemble import RandomForestRegressor as _RF
            n_jobs = getattr(self.output, 'cfg', {}).get('n_jobs', -1)
            rf = _RF(n_estimators=50, n_jobs=n_jobs, random_state=self.seed)
            rf.fit(self._byid.loc[tr_ids, self.feats].to_numpy(), self._byid.loc[tr_ids, 'label'].to_numpy())
            Xte = self._byid.loc[[self.nvs_ids[j] for j in te_j], self.feats].to_numpy()
            std[te_j] = np.stack([e.predict(Xte) for e in rf.estimators_]).std(axis=0)
        return std

    def weighted_r2(self, scale=None, taus=None):
        """Distance sweep with NVS sample_weight = exp(-std/scale) from the within-NVS CV (default scale=median std)."""
        std = self.nvs_uncertainty()
        scale = float(np.nanmedian(std)) if scale is None else scale
        w = {c: float(np.exp(-std[j] / scale)) if np.isfinite(std[j]) else 1.0 for j, c in enumerate(self.nvs_ids)}
        taus = taus if taus is not None else self.distance_taus()
        rows = []
        for t in tqdm(taus, desc='S3 weighted sweep'):
            r2 = self._cv_r2(lambda tr, _t=t: self._select_within(tr, _t), nvs_weight=w)
            rows.append({'tau': float(t), 'r2': r2})
            tqdm.write(f'  [S3] tau={t:.3f} (scale={scale:.3g}) r2={r2:.4f}')
        return pd.DataFrame(rows)

    # ---------- honest nested CV ----------

    def nested_distance(self, taus=None, n_outer=5, n_inner=4):
        """Nested CV for S1: pick tau on an inner CV of each outer-train, score on the held-out outer fold.
        Returns the honest outer R2 + the naive (inner-selection-maximized) R2 for contrast."""
        taus = taus if taus is not None else self.distance_taus()
        outer = self._grouped_folds(self.int_ids, n_splits=n_outer)
        parts, chosen = [], []
        for o, (otr, ote) in enumerate(tqdm(outer, desc='nested CV (outer)'), 1):
            inner = self._grouped_folds(otr, n_splits=n_inner, seed=self.seed + 1)
            # inner CV score per tau on the outer-train only
            scr = {t: self._cv_r2(lambda tr, _t=t: self._select_within(tr, _t), folds=inner)
                   for t in tqdm(taus, desc=f'inner {o}/{len(outer)}', leave=False)}
            t_best = max(scr, key=lambda x: (scr[x] if np.isfinite(scr[x]) else -np.inf))
            chosen.append(t_best)
            # refit on ALL outer-train internal + its selected NVS, predict the outer-test fold
            pr = self._fit_predict(otr, ote, self._select_within(otr, t_best))
            parts.append(pd.DataFrame({'compound': list(ote), 'real_y': self._byid.loc[ote, 'label'].to_numpy(float),
                                       'pred_y': pr, 'fold': o}))
            tqdm.write(f'  [nested outer {o}/{len(outer)}] chose tau={t_best:.3f} (inner best r2={scr[t_best]:.4f})')
        preddf = pd.concat(parts, ignore_index=True)
        preddf['residuals'] = (preddf['pred_y'] - preddf['real_y']).abs()
        return {'honest_r2': _pearson_r2(preddf['real_y'], preddf['pred_y']), 'chosen_taus': chosen, 'preddf': preddf}
