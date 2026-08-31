# ADME_ML — Project Wiki

Durable aggregate memory for this repo. Survives context compaction. Aggregate-only (no SMILES / compound IDs / per-compound values).

---

## Solubility cleaning sweep — thermo-only augmentation wins (2026-07-20, `python/solubility_cleaning_sweep.py`)

9 cleaning strategies for the public solubility augmentation, single-task RF, internal test FIXED
(cleaning touches public train only). Ranked by **R²_det** (coeff. of determination — calibration-sensitive;
squared-Pearson r² is affine-invariant and was masking everything). Verified leakage-free by a workflow
(0 internal↔public InChIKey twins, id-disjoint, NaN-free; asserts added). Output: `output/predictions_runs_solsweep/`.
**CV (calibrated) result:** thermo-only family beats all full-public strategies despite 6× fewer rows (18k vs 105k):
S7_combo R²det 0.727 / S9_thermo_pruned 0.720 / S2_thermo 0.711 vs contaminated baseline S0 0.652 (+0.075), RMSE_log 0.623 vs 0.703.
Thermo fixes calibration: bias ≈0 (vs −0.05..−0.08 full-public) and R²det_RAW +0.52..0.54 (vs ≈0). **Matching assay
type (internal is thermodynamic; 82.5% of public was kinetic) beats data volume** — the dominant lever. Physical-clip /
IQR / dedup alone barely move CV (kinetic majority remains); dedup slightly hurts (blends thermo+kinetic twins).
**pearson_r² is FLAT ~0.73 across all 9** → confirms selection must use R²_det, not Pearson. **Winner: S2 (plain thermo) promoted** (S9/S7 within noise; S2 has best raw calibration + ~0 bias, simplest).
Caveat: TEMPORAL arm is negative R²_det for all 9 (newest-30% is tiny + 55% floor-censored → no model beats the mean there);
CV is the decision basis.
**PROMOTED to pipeline (2026-07-21):** `SOL_PUBLIC_TYPES: [thermodynamic_experimental]` + `SOL_MAX_LOG10_UM: 6.0` clip +
`SOL_DROP_ORIGINS: []` knob (=[PHYS,AQUA] reproduces S9) in config; `build_solubility_features.py` applies clip+blocklist.
Rebuilt `public_solubility.parquet` 106k→**17,877 thermo-clean** rows (DS preserved; contaminated backup at
`public_solubility.contaminated_bak.parquet`). `run_RF` now pulls 17,877 public. **NOT yet redeployed** (MLTrail re-register)
nor chemprop-rerun on clean data — those are the follow-ups. NOTE: `run_RF`'s summary reports Pearson r² (blind to the
gain); the calibration lift shows only in the sweep's R²_det.

## ADME cleaning sweep — thermo-only approach generalized to the other 6 endpoints (2026-07-21, `python/adme_cleaning_sweep.py`)

Same design as the solubility sweep, for logd/hlm/mlm/rlm/mdck/ppb (caco2 excluded; solubility done). Public comes as
3 source classes per endpoint: **EXP** (experimental — TDC/Biogen), **NVS** (Novartis-NIBR predicted), **ADM** (ADMETlab
predicted). Cleaning touches public only; internal test FIXED; RF single-task temporal+CV; ranked by R²_det. Config knobs
under `ADME_CLEAN_SWEEP` (predicted_cap 40k for tractability, per-endpoint phys ranges, winsor/iqr). Strategies S0_all_raw /
S1_all_phys / S2_exp_only / S3_drop_adm / S4_best_combo / S5_internal_only (degenerate ones auto-aliased per endpoint).

**Source audit (modelling-space medians; `output/predictions_runs_admesweep/adme_source_audit.csv`)** — the dominant
lever is again assay/source matching, not value-clipping (physical clip triggers on only 5 rows total; **0 InChIKey leakage**
anywhere):
| ep | internal | EXP | NVS | ADM | note |
|----|---------|-----|-----|-----|------|
| logd | 3.10 | 2.36 | 2.84 | 3.26 | well-aligned (identity) — augmentation should help |
| hlm | 1.29 | **1.32** | 2.01 | – | EXP matches internal; NVS biased +0.7 |
| mlm | 1.40 | – | 2.34 | – | no experimental; NVS biased +0.94 |
| rlm | 1.07 | 2.06 | 2.39 | – | only **39 internal**; both sources biased +1.0–1.3 |
| mdck | −0.62 | – | 1.31 | 1.01 | no experimental; both predicted biased +1.6–1.9 (wrong assay variant) |
| ppb | −1.95 | −1.30 | −1.40 | −1.15 | our bRo5 bind more; all public shifted ~+0.6 |

Predicted public (NVS/ADM) carries a systematic upward bias vs our bRo5 internal (higher clearance-stable / lower Papp /
higher binding); experimental (esp. hlm) is far better calibrated. Hypotheses: mdck & mlm (predicted-only, badly shifted)
may prefer **internal-only** on R²_det (challenges current `mdck:[NVS]` deploy); hlm should prefer EXP; logd benefits from
augmentation. **DONE (2026-07-21)** → `output/predictions_runs_admesweep/adme_cleaning_sweep.csv`. Best strategy per endpoint, CV
R²_det (prev unswept = deployed all-source augmented):
| ep | prev R²det | swept R²det | winner → policy |
|----|-----------|-------------|-----------------|
| logd | 0.766 | 0.801 | internal-only (drop all public) |
| hlm  | 0.380 | 0.538 | internal-only (drop EXP+NVS) |
| mlm  | 0.455 | 0.610 | internal-only (drop NVS; no EXP exists) |
| rlm  | −0.149 | 0.234 | **experimental-only** (Biogen; drop NVS) |
| mdck | 0.088 | 0.479 | internal-only (drop NVS; both predicted shifted) |
| ppb  | 0.385 | 0.481 | **experimental-only** (drop NVS+ADM) |
(solubility handled by SOL_* thermo switch: 0.652→0.711.) **Every endpoint improved.** Verdict: **predicted public
(NVS/ADM) hurts every endpoint** (calibration-blind Pearson r² hid it — e.g. rlm prev Pearson 0.324 masked R²det −0.149);
augmentation only earns its keep with matched experimental (rlm's 39-cmpd internal, ppb) or assay-matched (solubility thermo).
The public-only→internal transfer arm confirms public alone transfers worse than internal everywhere (e.g. logd 0.59 vs
internal-CV 0.80). caco2 added later (CV: 0.262→**0.733** internal-only; predicted sources hurt it too). CV winners: solubility/rlm/ppb→[EXP],
logd/hlm/mlm/mdck/caco2→internal-only.

## Temporal validation + CV↔temporal reconciliation (2026-07-21, `python/temporal_eval.py`)

The single-cut newest-30% split is degenerate (newest-block variance collapses → R²det explodes). Replaced with **rolling-origin
(expanding-window) CV, disjoint next-block tests, predictions pooled over the newest 50%** — each compound scored once by a
past-only model; headline = **RMSE + bias (+ bootstrap CI)** since R²det stays degenerate on small/censored blocks. Arms per
endpoint: `previous_deployed` (config policy) vs `internal_only` vs `experimental`([EXP])/`predicted`. Also a fraction-sensitivity
curve and applicability-domain columns per pred_df. Global-vs-per-endpoint cut: chose per-endpoint percentiles (balanced blocks);
SRB id = registration-order proxy.

**Reconciled per-endpoint policy (CV winner vs temporal winner):** 6/8 agree.
| ep | CV R²det (prev→new) | temporal winner (RMSE) | reconciled |
|----|--------------------|------------------------|-----------|
| solubility | 0.652→0.711 | EXP-thermo 0.599 | **[EXP] thermo** ✓ |
| logd | 0.766→0.801 | all-source 0.778 (R²det 0.596, best temporal) | internal-only *(CONFLICT, low-conf; augment defensible)* |
| hlm | 0.380→0.538 | **EXP 0.528** (vs internal 0.643) | **[EXP]** *(FLIP: CV said internal-only)* |
| mlm | 0.455→0.610 | internal 0.605 | internal-only ✓ |
| rlm | −0.149→0.234 | EXP 0.608 (4/5 folds) | **[EXP]** ✓ |
| caco2 | 0.262→0.733 | internal 0.478 (R²det +0.55) | internal-only ✓ |
| mdck | 0.088→0.479 | internal 0.507 (R²det +0.20) | internal-only ✓ |
| ppb | 0.385→0.481 | EXP 0.428 | **[EXP]** ✓ |
**Two conflicts:** hlm flips to EXP (temporal is deployment-realistic; experimental clearly best out-of-time); logd ambiguous
(CV internal-only, temporal mild-augment — user's call). **Per-fold robustness is weak** (pooled winner wins only 2–4/5 folds;
only rlm 4/5 robust; logd/mlm/ppb 2/5 = toss-ups) → winners are directional. **Proposed `augmented_sources` (NOT yet wired,
awaiting confirmation esp. logd):** solubility/rlm/ppb/hlm → [EXP]; logd/mlm/caco2/mdck → []. Net: drop predicted NVS/ADM everywhere.

**Applicability domain — NN-distance predicts error (2026-07-21).** Per-compound `nn_tanimoto_dist` (1−max ECFP4 Tanimoto to
train) and `scaffold_novel` added to every temporal pred_df. Pooled over 652 newest-compound predictions (error standardized
per endpoint): **Spearman(NN-dist, |err|)=+0.34**, and mean standardized |err| rises monotonically 0.65 (dist≤0.3) → 1.31
(dist>0.7) — error ~doubles. **Novel scaffold |err| 0.89 vs known 0.51** (~75% worse). All 8 endpoints positive (strongest
mlm/solubility/logd; ppb flat, n=35). Explains the weak temporal metrics (newest blocks are 45–70% novel scaffolds) and gives a
deployable confidence gate: predictions within ~0.3 Tanimoto of train are ~2× more trustworthy than beyond 0.7.

## Temporal fraction-split RF-vs-Chemprop experiment (2026-07-21, `python/temporal_fractions_{rf,chemprop}.py`)

Overnight run to compare **RF single-task vs Chemprop multitask vs internal-only** on realistic temporal holdouts,
config `TEMPORAL_FRACTIONS`. Single-cut split per endpoint (train oldest 1−f, **test newest f**, by SRB id);
started with **f=0.3 (70/30)** only (full 3-way 0.4/0.3/0.2 deferred — the uncapped 273K NVS makes RF fits slow).
Chosen so we can pick a cut with real test spread (report `test_std`; e.g. caco2 newest-20% has std≈0 → R²det degenerate).

- **Arms** (config): RF `internal` + RF `augmented`; Chemprop `augmented` only for **sol,mdck (all8)** and **hlm,mlm,rlm (clearance)** — NOT logd/caco2/ppb. hlm & rlm run **both** EXP and EXP+NVS; mlm only NVS (no experimental MLM data). NVS is used by 6 endpoints (logd,hlm,mlm,rlm,mdck,ppb); solubility & caco2 use EXP only.
- **FULL public, NO cap** (the sweep's 40K cap was tractability-only; size experiment showed full ~2× temporal — see above). NVS=273,638, ADM=63,136.
- **Holdout = "virtual enumerated":** the target's newest-f internal compounds are removed from training ENTIRELY (all tasks); public rows whose InChIKey matches a test compound are dropped. Chemprop runs **per target endpoint** (only that endpoint's test held out) so its per-endpoint train matches RF single-task; multitask aux tasks stay internal-only.
- **Wide multitask:** predicted sources (NVS/ADM) keep ALL their covered-endpoint columns (real cross-task transfer); experimental (EXP) is single-endpoint. RF single-task uses target column only.
- **Custom Novartis-column cluster arms** (`python/build_novartis_clusters.py` → `tf_novartis_<name>.parquet`): internal target + specific Novartis pred columns as auxiliary tasks. **mdck_perm** = mdck + [LE-MDCKv2/v1 LogPapp, Caco-2 LogPapp, MDCK-MDR1 LogER, logPAMPA]; **ppb_fu** = ppb + [LogFu Rat/Human/Mouse/Dog/Monkey, HPLC LogFu HSA, LogFubrain, LogFumic, Direct NIBR LogP/LogD7.4]. (ppb has no grouping arm — cluster only.)
- **Chemprop arms vs RF:** LMs drop the EXP-only chemprop arm (RF keeps both); chemprop hlm/rlm = EXP+NVS, mlm = NVS. Config `endpoints[*].cp_arms` / `cp_clusters` separate from RF `augmented`.
- **Fast model (config `chemprop_hp`/`chemprop_epochs`):** hand-picked config {depth 3, msg-hidden 300, ffn 2×300, dropout 0.1464, agg sum}, 30 epochs, **ensemble=1**. (2026-07-22: tried HPO trial 0 [depth3/hidden300/ffn3×1200/dropout0.0727/agg norm] but it lost to hand-picked on these small temporal blocks — worse RMSE+R²det on 9/12 arms — so reverted. Beefy best remains trial 21 depth6/hidden1800 if accuracy>compute.)
- **Descriptors precomputed ONCE:** `tf_ds_cache.parquet` (440,061 smiles × 200 `DS_` RDKit2DNormalized) reused from the built parquets + internal; fed via chemprop `--descriptors-columns` (verified byte-identical to on-the-fly `v1_rdkit_2d_normalized`), so no per-run descriptastorus recompute of the 273K public rows.
- **Output:** `output/predictions_runs_temporal_fractions/<ep>/f30/{rf_internal,rf_<srcs>,cp_<srcs>,cp_<cluster>}.parquet` — pred_dfs (compound/real_y/pred_y MODELLING space + nn_tanimoto_dist/scaffold_novel vs INTERNAL train), loadable by `endpoint_metrics_table`.

## Per-endpoint summary-metrics tables in the notebook (2026-07-21, `vignettes/Summary_results.ipynb` cell "## Summary metrics")

`endpoint_metrics_table(pred_dict, endpoint)` (defined in the solubility summary cell, reused by all 8 sections) renders one
HTML row per prediction arm: `source` (provider), regression (R²det, R²pears-squared, RMSE, N), `n_train`, and the full
classification set (Accuracy/PPV/NPV/F1±/MCC) at `CUTOFFS[endpoint]` with the endpoint's convention sign. **Best per column =
bold cyan, worst = orange, scored WITHIN each block separately** (so the best temporal arm is visible even though temporal
values run lower than CV); CV arms sorted above a double rule, temporal arms below, each block sorted by R²det desc. The table
is emitted from the (merged) load cell — `<ep>_summary = endpoint_metrics_table(pred_<ep>, '<ep>')` appended after the loader. Exact
duplicate metric rows collapse into one label (e.g. `internal_cv = cv_internal (winner)`); the `cv_winner` alias is folded in and
marked. `n_train` + `source` come from **`python/adme_train_counts.py`** → `output/train_counts.csv` (counts only, no fits):
n_train = internal + cleaned-public pool (CV: all internal + per-strategy n_public matching the sweep logs; temporal: oldest-90%
window + pooled public capped at 40000, as temporal_eval does). Providers (aggregate origins): logd=AstraZeneca, hlm/ppb=Biogen+AZ,
rlm=Biogen, caco2=Wang(TDC), solubility=PharmaBench/ChEMBL/Biogen(+); NVS=Novartis-NIBR, ADM=ADMETlab (both predicted).

## Data-quality audit — public value distributions vs internal (2026-07-20)

Prompted by a raw-units RF run giving RMSE ~317,000 µM for solubility. Audited every source's value
distribution per endpoint (aggregate percentiles only). Findings:
- **solubility EXP is contaminated**: log10 max 8.63 → 4.26e8 µM = **426 mol/L** (impossible; water ~55 mol/L).
  **777 rows (0.73%) > 1 mol/L**, 98 > 10 mol/L. Sits 4+ log-units past internal max (4.25) → contaminates even
  the log model; blows up raw training. Likely a `harmonize_sol.py` µg/mL→µM unit bug (bad/missing MW) on a subset.
- **internal solubility is 55% floor-censored** at log10 0.19 (≈1.56 µM, detection limit) → solubility R²/RMSE inherently limited.
- **predicted-source (Novartis) tail outliers, physically implausible**: rlm max 76,600 µL/min/mg (p99.9 1,630),
  caco2 max 9,290 ×1e-6 cm/s (p99.9 153, realistic Papp ≲200), mdck max 880. Clip pseudo-labels before augmenting.
- **distribution shifts (not unit bugs)**: mdck NVS median 20.6 vs internal 0.24 (~85× higher; mdck augments NVS-only);
  ppb internal tightly bound (raw 1–19% unbound, median 1.1%) vs public spanning 0–100% → explains ppb transfer R²≈0.
- **clean**: logd (identity, all sources consistent), hlm, mlm.
Recommended (NOT yet applied): drop/clip solubility EXP >~log10 6 + fix harmonizer; clip NVS pseudo-labels to physical
ranges (CLint ≲5000, Papp ≲200); reconsider mdck NVS-only; flag solubility censoring. Applies to the MAIN analysis, not just raw.
Takeaway: modelling in raw units surfaced contamination that log space was masking (see the modelling-space metrics section below).

## Current objective

Train a public-data ML model to predict **thermodynamic solubility** of in-house compounds (PROTACs / molecular glues, bRo5).
- **In-house target:** `data/20260625_thermoSol.csv` — thermodynamic solubility in μM, range ~1.5 → 21,000 μM (4+ orders of magnitude → model in log space). Censored `< X` values parsed to float in `vignettes/Main.ipynb`.
- **Feature/compound reference:** `data/protacdb2.0_zinc_chembl_dataset.csv` (ProtacDB 2.0 + ZINC + ChEMBL, with predicted ADME columns — clearance, LogFu, LogP/D, permeability, CYP; **no solubility column**).
- **Strategy (2026-06-25):** pretrain on curated thermodynamic small-molecule sets → fine-tune on in-house μM data. Multi-task option using Biogen-Fang ADME endpoints.

## Sharing slides / files with the laptop (reverse SSH tunnel) — 2026-07-20

The cluster (`dl`, 192.168.146.108) has **no mount** to the user's Windows Desktop; VS Code Remote-SSH
is just a terminal channel. File transfer to the laptop rides a **reverse SSH tunnel** (same mechanism
MS_ML documents for Dropbox-over-SSH).

- **Laptop is WSL2 behind Windows NAT.** The user opens the tunnel from a **WSL terminal** (VS Code's
  *Windows* ssh client won't reach WSL's sshd, so a dedicated side-session is required; keep it open):
  `ssh -R 2222:localhost:22 -N gtamo@192.168.146.108`
- **Then from the cluster**, `localhost:2222` reaches back to the laptop (key-auth, BatchMode works):
  `scp -P 2222 -o BatchMode=yes -o StrictHostKeyChecking=accept-new <file> gtamo@localhost:/mnt/c/Users/gtamo/Desktop/GT/Claude_ppt/`
  WSL `/mnt/c` = Windows `C:`, so this lands in `C:\Users\gtamo\Desktop\GT\Claude_ppt\` (default drop dir).
  `mkdir -p` the dest first; verify with `ssh -p 2222 gtamo@localhost 'ls -la <dest>'`.
- **If port 2222 refuses** → the tunnel isn't up; ask the user to (re)open the WSL side-session.
- **Privacy:** only transfer aggregate/non-sensitive artifacts (slides of counts, metrics, source names —
  never SMILES/structures/per-compound values). Slides are built to `output/ppt/`
  (`python python/make_datasets_slide.py`, python-pptx in the `ML` env). Verified live 2026-07-20.

## Unit tests for the systematic runners (`tests/`, 2026-07-20)

`unittest` (stdlib — no pytest dep, mirrors MS_ML). Run in env `ML` from repo root:
`python -m unittest discover -s tests` (**22 tests, ~57s**, 1 skipped). `tests/_adme_fixture.py` builds a
~1K-compound **LOCAL subset** of the real cache into a temp dir (first-batch parquet slice, values never
printed) and points each runner's module globals (`CACHE`/`FEATDIR`/`output_dir`) at it — no production
output touched; temp dirs auto-cleaned.
- **`test_rf_systematic.py`** (14): features load (4269), `internal_ep` labels; **no-leakage** for all 3
  arms — temporal (train∩test=∅, test⊆newest-30%), CV (OOF = internal only, public never scored, each
  internal cmpd predicted once), public-only (train/test compound-disjoint); **MLTrail** register→predict
  (3/3 public SMILES non-null, records unit/features_type), predict invariant to row/col order + extra
  columns, **idempotent re-registration** (same id, new version); **source policy** (config override
  filtered to available; solubility excludes predicted ADM even when present — poisoning guard);
  **modelling_unit** transform→label contract (identity/log10/logit_pct); **CV determinism** (same seed).
- **`test_chemprop_systematic.py`** (8): chemprop temporal test == RF temporal test (bit-identical);
  `build_combined`/`build_public_only`/CV-fold leakage; all-8 target columns present even for source-less
  endpoints; split determinism; eval+scoring via a **mocked chemprop CLI** (perfect preds → r²≈1, OOF
  parquet written) incl. the endpoint-named vs `pred_i` column branch; MLTrail chemprop register→predict
  (mocked CLI, invalid SMILES→null). `TestRealTrain` (real 1-epoch CV) skipped unless `RUN_CHEMPROP=1`.
Chemprop training is never run in the fast suite — all leakage logic lives in the pure split builders.

## Metrics are computed in MODELLING space, not raw assay units — deliberate (2026-07-20)

Targets are transformed from raw CDD values before modelling (`python/extract_adme.py::_transform`):
**log10** for solubility/hlm/mlm/rlm/caco2/mdck, **logit** of fraction-unbound for ppb, **identity** for
logd. So `internal_targets.parquet`, training, predictions and all reported R²/RMSE live in log/logit
space. Checked whether recomputing R² in RAW units changes model selection (best model per endpoint,
temporal): logd unchanged (identity, sanity ✓); clearance/ppb move ≤0.04; **caco2 +0.19 (→0.94), mdck
+0.17 (→0.61)** and the argmax flips for caco2 (RF int→aug) and mdck (CP all8→RF int). These flips are
**leverage artifacts** — raw-space Pearson r² is dominated by a few high-Papp compounds (mdck's CP:all8
has Spearman only 0.149 yet raw r²≈0.61). Spearman (rank-invariant) doesn't change. **Decision: keep
selection + reported metrics in modelling space; back-transform ONLY the delivered predictions** to raw
units for chemists (`vignettes` MLTrail-predict cell inverts per `ADME_ENDPOINTS[ep]['transform']`:
log10→10**p, logit→100/(1+10**-p), identity→p).

## ADME_build_ML.py — notebook↔module port (2026-08-20, in progress)

`python/ADME_build_ML.py` mirrors `MS_build_ML.py`: `PARAMS`/`DATA`/`OUTPUT` scaffold, runs standalone
(`--config`, `--overwrite`) AND callable from `vignettes/Multitask_adme_preds.ipynb` (`%autoreload 2`).
Section 0 ported: `params = PARAMS(cfg).load_params()` (+ uppercase-key shim), `data = DATA();
data.load_df_all(params)` -> `data.df_all` (326×102, cached `data/20260707_all_adme.csv`, overwrite re-pulls
CDD). Notebook keeps bare-name shims (`df_all`, `dfs`, `MF_features`, `ML_data` = the `data.*` objects).
Solubility analysis flow to port next: harmonize -> H237 features -> ML frame -> power-set transfer / CV /
temporal (internal vs Biogen-Fang augmented; regression + 15µM classification, log & raw).

**Per-endpoint eval flow internalized (2026-08-25).** `DATA`: `get_internal_public_sets(k, min_n=1000)` ->
`self.k/d/internal/pub/origins` (InChIKey leak-drop); `select_best_combo_and_update(params, k)` ->
`self.combo/pub_ids/int_ids/fold_ids/fold_ids_aug/train_temp_ids/test_temp_ids` (combo = `BEST_PUBLIC[k]`
origins present, else all); `transfer(combo, make_model, use_cuml)` -> one combo's ext->internal metrics.
New `OUTPUT(params)` owns the model machinery + results: `make_model(use_cuml, n_bins)`,
`predict_and_record(d, result_df, exp_name, ids, col_to_rm, use_cuml)`, `run_origin_powerset(data)` ->
`output.res_origin_powerset`, `assess_predictions(data, outpath)` -> `output.metrics_results[data.k]`
(6 arms: ext->internal, internal/augmented CV, internal/augmented temporal, ext->temp; compute-or-load
pickle at outpath). Notebook: `output = OUTPUT(params)` in the data cell; the **sol** section now calls these
methods (other 7 sections still inline until harmonized); the metrics HTML table still renders in the notebook
via `endpoint_metrics_table_from_dict(output.metrics_results[k], k)`.

**CLI + MF cleanup (2026-08-25).** `python python/ADME_build_ML.py --config … --assess_RF_all_endpoints`
[--endpoints a,b] [--min_n N] builds `df_all` + `MF_features['all']` then runs
`OUTPUT.assess_all_endpoints` -> one pickle per endpoint at config `METRICS_PKL_DIR`
(`output/results/20260825_metrics/`). Per-endpoint feature builders removed
(`build_MF_features_{solubility,logd,mdck}`, generic `_endpoint_MF`, and the alias) — features now come
only from `MF_features['all']` via `DATA._feats(k)` (falls back to 'all' when no per-endpoint matrix).
`get_<k>_data` and the whole-set `build_MF_features(params)` KEPT (`load_combine_dfs` calls `get_<k>_data`
to build `dfs[k]`). **Update (2026-08-25): `get_<k>_data` is now ALSO fully generic** — see below.

**Deploy to MLTrail (2026-08-26, `OUTPUT.deploy_endpoint` / `deploy_all_endpoints`, CLI `--deploy_RF_all_endpoints`).**
Fits the deployable RF for each endpoint on **internal + BEST_PUBLIC public** (all rows, no held-out fold) and
registers a **NEW** MLTrail model `adme_<k>_h237` (the H236 champions `adme_<k>` stay untouched). Config knobs in
`DEPLOY:` — `features_type: H237`, `experiment_suffix: _h237`. Confidence = **conf_recal**: the method runs the
**augmented CV** (`fold_ids_aug`, public in TRAIN only, `uq=True`), calibrates via `ML_Reg.calibrate_confidence_params`,
and stores `{rmse_cv, recal_a, recal_b, label_std}` **inside the model bundle artifact**
(`{model, feature_cols, endpoint, features, sources, n_train, transform, unit, calibration, sklearn_ver}`) — travels
atomically with the fitted estimator, versioned by MLTrail, read by the webapp in its one startup `joblib.load`. The
recal scalars are ALSO mirrored into the registry `metrics` dict so `registry.details()` exposes them without loading
the artifact. Features = **H237** (4469; `MF_features['all']`); MLTrail's built-in `H237` featurizer re-derives them at
predict time (**needs `descriptastorus` in the predict env — present in `ML`**). Run:
`python python/ADME_build_ML.py --config config/config.yaml --deploy_RF_all_endpoints [--endpoints a,b] [--dry_run]`.
Tests: `tests/test_deploy.py` (synthetic, fake registry — no vault write).

**CLI runner (2026-08-27, `python/run_nvs_cellab.py`).** Runs Cell A+B headless with crash-safe incremental saves
(SSH-drop cost the user 40 min of interactive compute). `python python/run_nvs_cellab.py --config … --outdir
output/results/20260827_NVS_cellab --levers baseline,s1,s5,s3,s2,nested [--n_taus N] [--taus a,b] [--resume]`.
Writes per arm the OOF `<arm>_preddf` + `<arm>_metrics` into `metrics_dict.pkl` (consumable by the notebook's
`endpoint_metrics_table_from_dict(d,'mdck')` — arm keys avoid 'temp' so all land in its CV category), each pred_df
to `preddfs/<arm>.parquet`, and full training-set membership: `folds.pkl` (internal CV folds), `nvs_fold_distance.parquet`
(NVS train at τ = distance<τ per fold), `nvs_scaffold_groups.parquet`+`arms.json` (S2), `nvs_weights.parquet` (S3),
`train_membership.pkl`. Every arm streams to disk the instant it finishes; `--resume` skips arms already present.
Module gained `cv_preddf` (returns the OOF frame) and `nested_distance` now returns its `preddf`; both tested. Run
detached (screen/nohup). NVS pool for mdck is large — `all_nvs` + sweeps are the heavy part; start `baseline,s1`.

**Interim mdck result (2026-08-27, ~partial run).** internal=107, NVS(Novartis-NIBR)=273,241; NVS→internal distance
is broadly far (10th pct ≈ 0.768, i.e. max Tanimoto ≈ 0.23). Negative transfer confirmed and it is **bias, not rank**:
`r2` (Pearson²) barely moves across arms (~0.44–0.49) while **R²det collapses** internal-only 0.474 → all-NVS 0.134
(RMSE 0.429 → 0.551 log10). Distance lever monotone: less/closer NVS = less damage; the nearest ~1.3k (τ=0.70,
sim>0.30) nudges r2 to 0.493 (>0.475) but R²det 0.403 still < internal-only — a tight near-shell helps ranking,
not the bias-sensitive metric. So distance filtering cannot beat internal-only; **bias-correction is the lever that
targets the actual failure mode.**

**mdck UPDATE — a tight near-shell BEATS internal-only; bias-correction refuted (2026-08-27).** Re-running S1 on a
tight grid found the sweet spot the coarse grid (≥0.768) had missed: R²det vs τ is an inverted-U peaking at
**τ=0.60** (nearest ~101 NVS): internal-only 0.474 → **0.528** (RMSE 0.429 → 0.407, r2 0.475 → 0.536). τ=0.50 0.497,
τ=0.65 0.501, τ=0.70 0.403, τ=0.768 0.231, τ=1.0 0.140. So a small raw near-shell of Novartis (~100 close analogs,
sim>~0.4) genuinely improves mdck — distance filtering, not bias-correction, is the winning lever (my earlier
"distance can't beat internal-only" was a coarse-grid artifact). **`shift` bias-correction HURTS** (τ0.6 0.478, τ0.65
0.109, τ0.70 −0.085): matching medians conflates true population difference with label bias and stamps a large wrong
offset — the close labels are already usable raw, global recalibration corrupts them. `affine` behaves the same
(τ0.5 0.483, τ0.6 0.404, τ0.65 0.155, τ0.7 −0.203, all −0.221) — both corrections refuted. S1 window τ0.5–0.65 ALL
beat internal-only (0.497/0.528/0.501); peak τ=0.60. **Not yet banked:** τ=0.60 was selected on the same 107
compounds; validate with nested CV (`--levers nested --taus 0.5,0.55,0.6,0.65,0.7`) — adopt "internal + nearest-shell
NVS (dist<~0.6, ~100 cmpd, RAW)" only if honest R²det ≳ 0.50; also get a fold-spread/bootstrap error bar (effect ~+0.05).

**Model-family speed/accuracy benchmark (2026-08-27, `python/bench_models_transfer.py`; transfer: train all 273k NVS
→ predict 107 internal mdck).** fit-time | R2_pears | R2det: **LightGBM 33s | 0.350 | −3.33**; RF-50 110s | 0.412 |
−3.72; RF-200 (champion) 440s | 0.407 | −3.57; **ElasticNet 1397s | 0.283 | −1.46**. Conclusions: **LightGBM is ~13×
faster than champion RF**, nonlinear, near-RF ranking → the high-throughput screening learner for the subset search
(RF-50 a middle option; confirm finalists with RF-200). **ElasticNet is OUT** — slowest AND least accurate on the real
matrix: correlated fingerprint features make coordinate descent converge slowly (the synthetic-data ~49s estimate was
misleading; real 23min). All R2det strongly negative = pure NVS→internal transfer is hopeless (re-confirms NVS is only
useful as near-shell augmentation, not standalone). RF's 200 trees are overkill (RF-50 ≈ RF-200 on ranking, 4× faster).
`lightgbm==4.7.0` added to requirements.txt.

**NVS instance-selection campaign (2026-08-27, `python/nvs_campaign.py`, unattended batch).** Autoresearch-style
menu search for the NVS subset that maximises internal R2det, LightGBM workhorse + champion-RF cross-check, honest
nested-CV verdict. ~22 literature-grounded strategies (instance selection + importance weighting vs negative transfer;
NN + classifier density-ratio weighting): baselines, distance-shell `dist_<tau>`, kNN local `knn_<k>`, similarity/exp
weighting `distw/expw`, classifier density-ratio `clfw`, support/range matching `range_*`, agreement/pseudo-label filter
`agree_*`, within-NVS uncertainty filter `lowunc`, outlier removal `noout`, combos. Each scored by grouped-CV R2det
(InChIKey folds, leakage-free); **nested CV** picks best-on-inner, scores on held-out outer (the trusted number, printed
with a BEATS/does-not-beat internal-only verdict); huge-subset strategies excluded from nested (`--nested_max_nvs`).
`NVSSubsetSearch` gained a `learner` switch (`rf`/`rf50`/`lgbm`, via `_make`) and float32 fits (halves RAM on 273k).
Crash-safe/resumable; outputs `campaign_<ep>.csv`, `rf_check_<ep>.csv`, `summary_<ep>.json`, per-strategy preddf parquets.
Run detached (screen). Tested on synthetic data (all strategy families + nested). Real mdck run: pending.

**S5 bias-correction upgraded (2026-08-27).** The old `_bias_label` (affine on near-neighbour anchors, sim0=0.5) was
a no-op for mdck (no NVS that close). Rewrote `_bias_label(method, sim0, min_anchors)` — per-fold, train-only
(leakage-free): **`shift`** (default; match added-NVS MEDIAN to internal-train median — targets the constant offset,
always computable), **`affine`** (match median + IQR — offset + scale), **`anchor`** (near-neighbour affine, falls
back to `shift` when < min_anchors). Runner flag `--s5_methods shift,affine`; arms `S5<method>_tau*`. Decision metric =
whether any S5 arm's R²det clears the internal-only 0.474 line. Tests: `test_bias_correction_recalibrates_to_internal_scale`.

**Webapp switched to H237 + conf_recal (2026-08-26, `webapp/app.py`).** `_discover_champions` now gathers `adme_<ep>`
sklearn models keyed by features_type and **prefers H237 over H236 per endpoint** (so it works before/after deploy
completes). It `joblib.load`s each bundle for `model` + `feature_cols` + `calibration` in one read (`load_model`
dropped the extra keys). Confidence via new `_confidence(std, c)` = **conf_recal** `exp(-clip(recal_a+recal_b*std,0)/rmse_cv)`
when a calibration is bundled, else the training-label-std fallback `exp(-std/sigma)`. Featurizer switched to **H237**
(its columns cover H236 fallbacks, so one featurizer serves both; needs `descriptastorus`, present in `ML`). Eyebrow +
docstring updated. `_confidence` unit-tested (bounded, monotone, matches closed form, both fallbacks). Restart the app
after deploy finishes so it picks up the new H237 models.

**build_ML_data fully generic (2026-08-25).** The bespoke `build_ML_data_{solubility,logd,mdck}` are gone;
all 8 route through `DATA._endpoint_ML(endpoint)` (aliased for every endpoint). The only per-endpoint
difference was the solubility label cap — now config-driven: `ADME_ENDPOINTS[ep]['label_cap_raw']` (raw
units, applied in modelling space via the transform); solubility=15000 µM, others uncapped. `_endpoint_ML`
reads `self.params` (stored in `load_df_internal_exp_all`). Tests: `tests/test_build_ml_data.py` (synthetic
data — cap, feature merge, NaN/inf drop, SMILES dedup-prefers-internal, all-8-aliases).

**get_<k>_data also fully generic (2026-08-25).** The bespoke `get_{solubility,logd,mdck}_data` are gone;
all 8 route through `DATA._endpoint_dfs` (internal transform via `_TF`, cell-line `filter`, public-file
concat, `raw`=inverse). Added their public files to config `ENDPOINT_PUBLIC_FILES` (solubility:
[public_solubility], logd: [public_logd], mdck: [public_novartis_mdck, public_admetlab_mdck]). The alias
loop now sets BOTH `get_<ep>_data` and `build_ML_data_<ep>` for all 8. So the ONLY per-endpoint bespoke
code left is config (transform/unit/filter/label_cap_raw + the file lists). Test added for `_endpoint_dfs`
(transform/filter/multi-file-concat/raw).

**caco2 temporal split is degenerate (2026-08-25).** The newest 20% of INTERNAL caco2 compounds (the
temporal test slice, SRB-id order) all share ONE identical measured value (`test_unique_labels=1`), so
`real_y` is constant and Pearson/R²/linregress are mathematically undefined — this crashed
`assess_all_endpoints` at the caco2 `internal_temp` arm. The split LOGIC is correct; caco2's newest block
is genuinely constant (other 7 endpoints fine). Fix: `ML_Reg.get_reg_metrics_from_preddf` now guards
constant/`n<2` inputs — returns `nan` for pearson_r/r2/r2det/spearman_rho (RMSE/MAE still computed) instead
of raising. So caco2's temporal arms report `nan` correlations (honest); the run completes.

## Systematic deployment runs — vignette scripts (2026-07-18)

Two `PARAMS/DATA/OUTPUT/MAIN` runners in `python/` (user's preferred class shape) built this session,
scored on a **LOCAL per-endpoint temporal split** (newest 30% of *each endpoint's* compounds — fixes
the global split's solubility 29/91 train/test skew; also makes rlm/caco2 scorable). H236 features only
(DS ablation: mean Δ ≈ +0.018 r², mixed sign → not worth it; deployable via MLTrail's H236 featurizer).

**`run_RF_SingleTask_systematic.py`** (env `ML`) — champion RF, 8 endpoints × 4 arms (internal/augmented
× temporal/CV), pred_dfs → `output/predictions_runs/<ep>/`, registers deployable H236 models to MLTrail
(`adme_<ep>`). H236 cache in `output/features/`; `--resume` skips done endpoints. **Final temporal r²
(augmented, best datasets):** caco2 0.856 · mdck 0.321 · ppb 0.316 · logd 0.309 · mlm 0.221 · rlm 0.057 ·
sol 0.048 · hlm 0.002. **CV r² (robust):** logd 0.81 · sol 0.73 · mlm/rlm ~0.55/0.32 · caco2 0.55 · etc.

**Augmented-source policy (config `RF_SINGLETASK.augmented_sources`, config-vetted per endpoint):**
solubility=EXP · logd=EXP+NVS+ADM · hlm=EXP+NVS · mlm=NVS · rlm=EXP+NVS · caco2=EXP+NVS+ADM ·
**mdck=NVS · ppb=EXP+NVS**. Decided by:
- **ADMETlab (ADM) is PER-ENDPOINT, not uniformly bad** (2026-07-18 ablation, augmented-temporal): drop ADM
  → mdck **0.163→0.321**, ppb **0.149→0.316** (ADM HURTS); but ADM HELPS logd (0.31, −0.07 if dropped) and
  caco2 (0.86, −0.11 if dropped) → kept. solubility already EXP-only (ADMETlab predicted poisons it).

**`run_Chemprop_SystematicGroups.py`** (env `chemprop`, GPU) — multitask counterpart, **augmented+temporal
only** (no CV/internal — too small for a DNN). 3 groupings {sol,logd},{hlm,mlm,rlm},all-8 (autoresearch
best), each scored on the **same RF local-temporal tests** (leakage-safe: hold out the UNION of members'
tests, score each endpoint on its own newest-30%). Reads the SAME config-vetted augmented sources + the
HPO's `best_config.json` hyperparams at run time. Saves model dirs → `output/chemprop_models/<grp>/`;
**MLTrail deferred** (v1 can't predict chemprop — `backends.py` stub). Guarded launcher
`run_chemprop_groups.sh` waits for the HPO to fully exit, then runs on the finalized datasets.

**FINAL head-to-head (2026-07-19, DONE), same local-temporal tests, in-house Pearson r²** — RF internal-only
· RF augmented · Chemprop(best grouping) → deploy:

| endpoint | n | RF internal | RF augmented | Chemprop (best grp) | deploy → r² |
|---|---|---|---|---|---|
| solubility | 36 | 0.011 | 0.048 | **0.484** (all8) | Chemprop-all8 → 0.484 |
| logd | 96 | 0.141 | **0.309** | 0.288 (all8) | RF-aug → 0.309 (~tie) |
| hlm | 98 | 0.015 | 0.002 | **0.217** (clearance) | Chemprop-clearance → 0.217 |
| mlm | 98 | 0.023 | 0.221 | **0.323** (clearance) | Chemprop-clearance → 0.323 |
| rlm | 12 | 0.068 | 0.057 | **0.148** (clearance) | Chemprop-clearance → 0.148 |
| caco2 | 29 | **0.879** | 0.856 | 0.608 (all8) | RF-internal → 0.879 |
| mdck | 33 | 0.352 | 0.321 | **0.444** (all8) | Chemprop-all8 → 0.444 |
| ppb | 21 | 0.118 | **0.316** | 0.066 (all8) | RF-aug → 0.316 |

**Chemprop wins 5 (sol, hlm, mlm, rlm, mdck), RF wins 3 (logd, caco2, ppb).** Multitask sharing rescues
the fragile endpoints RF couldn't touch (sol 0.05→0.48, hlm 0.00→0.22 — biggest wins of the project).
caco2 is the only endpoint where augmentation doesn't help temporally (RF internal 0.879 > aug 0.856).
**Grouping matters:** hlm/mlm/rlm want the 3-task *clearance* block (collapse to ~0 in all8); sol/logd/mdck
want *all8*; the 2-task *sol_logd* grouping is worst for both its members (don't ship). **Deploy stack:**
clearance→chemprop-clearance · sol/logd/mdck→chemprop-all8 · caco2/ppb→single-task RF.

**Chemprop CV (2026-07-19, RUNNING)** — added a `--cv` path to `run_Chemprop_SystematicGroups.py`
(multitask analog of RF `eval_cv`): one shared **compound-level** K-fold split per grouping (a compound's
graph can't split train/test), **public always in train**, pooled out-of-fold predictions per endpoint →
`pred_<grp>_cv.parquet` + `summary_chemprop_cv.csv`, CV r² directly comparable to RF `internal_cv`/
`augmented_cv`. Running the 2 winning groupings via `run_chemprop_cv.sh` (detached): **clearance first
(~10h), then all8 (~20h)**; 5 folds each, same tuned HPO hyperparams + 50 epochs as the temporal run. Fold
models are throwaway (`persist=False`); the deploy artifacts remain the temporal `output/chemprop_models/`.

**Chemprop MLTrail deployment (2026-07-19) — NO LONGER DEFERRED.** MLTrail now predicts chemprop natively
(backend built + tested in the MLTrail repo — shells out to a chemprop CLI, works cross-env; see MLTrail
wiki 2026-07-19). Register the saved grouping models with `run_Chemprop_SystematicGroups.py --register`
(framework=chemprop, model_type=multitask_regression, target_columns=endpoints; idempotent `adme_mt_<grp>`;
runs a public-SMILES deploy sanity). Predict via `registry.predict(id, df, smiles_column=...)` → one column
per endpoint (modelling space, same as RF — no inverse transform). Smoke-verified end-to-end on the real
clearance model (hlm/mlm/rlm) from the `ML` env, CPU, ~6.5s. Not yet registered into the production vault
(awaiting go-ahead + which groupings).

**Metric panel sweep (2026-07-20) — temporal Pearson-r² was masking miscalibration.**
`python/compute_metrics_sweep.py` computes Pearson-r² + **R²_det (coeff of determination)** + RMSE + MAE +
Spearman + `calib_gap`(=Pearson-r²−R²_det) over all 63 saved pred_dfs → `output/predictions_runs/metrics_all.csv`.
Findings (settled):
- **CV is calibrated, temporal is NOT.** Every CV arm has calib_gap≈0 (R²_det≈Pearson-r²: e.g. logd
  internal_cv 0.809/0.801, sol 0.703/0.702). Temporal arms have hugely NEGATIVE R²_det (worse than the
  mean) — caco2 aug_temporal Pearson 0.856 but **R²_det −0.56, RMSE 0.725**; rlm temporal R²_det −120 (n=12).
  The small-n temporal Pearson-r² is correlation on a handful of points with terrible calibration → **trust
  CV for selection, not temporal.**
- **caco2 ADM decision FLIPS.** With-ADM aug_temporal is miscalibrated (R²_det −0.56, RMSE 0.725); **no-ADM
  is genuinely good (R²_det +0.607, RMSE 0.364, ≈ its CV 0.400/0.733).** The 0.856 Pearson was bias masked
  as correlation. → **caco2 should DROP ADM** (join mdck/ppb). logd holds up WITH ADM on every metric — keep.
- **Public-only is rank-good, calibration-poor:** transfers in Spearman (0.65–0.76) but negative R²_det
  (hlm clearance_publiconly −0.63 at Pearson 0.386) — absolute predictions on our compounds still need internal.
- Convention going forward: report R²_det + RMSE alongside in-house Pearson-r²; a large calib_gap = the
  "correlated but biased/flattened" failure.

**Public-only → internal experiment (2026-07-19, BUILT — run after CV).** External-validation /
domain-transfer baseline: train each endpoint's **winning** model on **public data ONLY** (zero internal
compounds in train) and predict **ALL** internal compounds. Isolates pure public→internal transfer (the
floor internal data lifts us above; every other run keeps internal in train). Arms mirror the deploy split:
- RF: `run_RF_SingleTask_systematic.py --public-only [--endpoints ...]` → `eval_public_only` (train=public
  ids, test=all internal via `K_fold_by_defined_IDs`) → `summary_public_only.csv` + `pred_public_only.parquet`.
- Chemprop: `run_Chemprop_SystematicGroups.py --public-only --groupings clearance all8` → `build_public_only`
  (all internal→test, public→train+10% val; verified 0 internal leakage) → `summary_chemprop_publiconly.csv`.
- Combine: `combine_public_to_internal.py` picks the winner per endpoint (config `PUBLIC_TO_INTERNAL.winners`)
  → `summary_public_to_internal.csv` (publiconly_r2 vs deploy_temporal_r2 = the transfer gap).
Launcher `run_public_to_internal.sh` waits for CV, runs both arms + combine. **Winners map = current
temporal head-to-head; REFRESH from CV before trusting the combined table.** Same tuned HPO hp + 50 epochs.

**RESULTS (2026-07-20, public→internal r², winner per endpoint, test = ALL internal):** logd 0.708 (RF) ·
caco2 0.536 (RF) · mlm 0.494 · hlm 0.386 · rlm 0.368 (clearance) · mdck 0.199 · solubility 0.181 (all8) ·
ppb 0.006 (RF). **Two tiers:** (a) public data alone already transfers well — logd + the clearance panel
(hlm/mlm/rlm) + caco2 (r² 0.37–0.71 with ZERO internal compounds); (b) genuinely need internal — ppb
(0.006, public useless), solubility (0.18), mdck (0.20). **CAVEAT:** public-only is scored on ALL internal
(n=317/324/…), the temporal deploy number on the newest-30% (n=96/98/…) — different/harder test, so
public-only ">" temporal for logd/clearance is largely a test-set artifact, NOT evidence internal hurts.
Clean with-internal reference = CV (also pooled over internal); recompute the gap vs CV when CV lands.

**Chemprop HPO** — `chemprop_hpopt.py` + `run_hpopt.sh`: Optuna TPE (offline, no ray/telemetry), stopped
early at trial 29/40 (plateaued 2026-07-19). **Best = trial 21, val_loss 0.36 vs stock-default 0.45 (−20%).**
Tuned config is much larger + regularized vs the autoresearch stock defaults: depth 6 (vs 3), message-hidden
1800 (vs 300), ffn 2×1200 (vs 1×300), dropout 0.15 (vs 0.0), batch 256 (vs 64), sum aggregation (vs norm).
`best_config.json` feeds the grouping run. `torch==2.13.0+cu126` for the A6000 GPU.

## Multitask ADME regression (2026-07-07, `autoresearch/predict_adme/`)

Extension of the solubility exercise to the full in-house ADME panel as **regression**. Goal:
per-endpoint single-task models + **one masked multitask Chemprop** over all endpoints, systematically
testing which public data helps. Features: MF fingerprints (`rdkit_tools.compute_H236_features`) +
Descriptastorus (`RDKit2DNormalized`). HARD RULE enforced: SMILES never viewed/printed.

**In-house data:** `data/20260707_all_adme.csv` (326 cmpd, CDD export). Endpoint spec in
`config/config.yaml → ADME_ENDPOINTS`; extracted by `python/extract_adme.py` to modelling space.
Compound id = `name` (SRB-XXXXXXX; digits = temporal recency). 8 endpoints (transform, n non-null):
solubility (log10 µM, 120) · logd (identity LogD7.4, 317) · hlm (log10 CLint, 324) · mlm (log10, 324) ·
rlm (log10, 39) · caco2 (log10 Papp A→B, 95) · mdck (log10 Papp A→B, 109) · ppb (logit of fu, 69).

**R² convention (2026-07-07): all reported R² = in-house `Statistics_tools.rsquared` = squared Pearson r
(offset/scale-invariant, always ≥0).** sklearn coeff-of-determination is kept as `r2_det` (penalizes
calibration, can go negative) for transparency. The two diverge sharply on the temporal split: strong
*correlation* (high Pearson r²) can coincide with poor *calibration* (very negative r2_det).

**Internal-only regression baseline — the metric to beat** (champion RF n200/md20/mf0.3/msl2, MF features;
`baseline_eval.py`). R² = Pearson r² (r2_det in parens where it diverges):
| endpoint | rand5 R² | rand5 Spearman | temporal R² | temporal Spearman |
|---|---|---|---|---|
| logd | 0.804 | 0.889 | 0.119 (det −1.11) | 0.475 |
| caco2 | 0.734 | 0.853 | **0.876** (det 0.227) | 0.578 |
| solubility | 0.706 | 0.777 | 0.012 (det −2.56) | 0.358 |
| mlm | 0.630 | 0.802 | 0.059 (det −10.4) | 0.206 |
| hlm | 0.567 | 0.705 | 0.004 (det −11.2) | 0.042 |
| mdck (MDR1) | 0.482 | 0.504 | 0.341 (det 0.294) | 0.369 |
| ppb | 0.435 | 0.611 | 0.088 (det −0.43) | 0.274 |
| rlm | 0.077 | 0.313 | 0.072 (det −174, n=12) | 0.054 |

- **Interpolation works** everywhere except rlm (n=39 too small): rand5 Pearson r² 0.44–0.80, Spearman 0.4–0.89.
- **Extrapolation, re-read under Pearson r²:** the earlier "catastrophic collapse" was a *calibration* failure
  (r2_det −9 to −174), NOT a correlation failure. By Pearson r², caco2 extrapolates well (0.88), permeability/mdck
  moderately (0.34), while clearance (hlm 0.004 / mlm 0.059 / rlm 0.072) and sol/ppb have genuinely **weak temporal
  correlation** (Spearman ≈0.04–0.27). So the newest-30% chemistry mainly breaks clearance *ranking*; where models
  keep the ranking (caco2/logd/mdck) only the absolute scale drifts. → public data still most needed for clearance.
- Caches: `internal_targets.parquet`, `internal_MF.parquet` (325×4270; 1 dup SMILES dropped). Descriptastorus
  (`internal_DS.parquet`) pending `pip install chemprop descriptastorus` into env `ML` (added to requirements.txt).
- Public-data search: deep-research run `wf_56ecaf77-357` (TDC ADMET group, Biogen-Fang, ChEMBL, AZ, Polaris).

**Public-data → in-house unit mapping (2026-07-07 deep-research, all HIGH-confidence/verified).**
Per-endpoint best public source and whether it merges directly, needs conversion, or has no matching-unit source:
| endpoint | public source | n | public unit | → in-house transform | tier |
|---|---|---|---|---|---|
| logd | Lipophilicity_AstraZeneca (TDC) | 4200 | logD@7.4 | as-is (identity) | **direct** |
| hlm | Clearance_Microsome_AZ (TDC) | 1102 | mL/min/g ≡ µL/min/mg | log10, no numeric change | **direct** |
| caco2 | Caco2_Wang (TDC) | 906 | log10 Papp (cm/s) | +6 (cm/s→1e-6 cm/s) | convert |
| ppb | PPBR_AZ (TDC) | ~1614 | **%bound** | fu=(100−%b)/100→logit | convert |
| hlm (aux) | Biogen-Fang HLM | ~3087 | **mL/min/kg** (bw-scaled) | ÷(MPPGL·LW) physiol. scaling→log10 | convert⚠️ |
| rlm | Biogen-Fang RLM | ~885+ | **mL/min/kg** (bw-scaled) | ÷physiol. scaling→log10 | convert⚠️ |
| mlm | — none — | | | multitask transfer from hlm/rlm only | **no direct** |
| mdck | — Biogen is efflux ratio (B-A/A-B), not Papp — | | | pretrain/secondary only; Papp unrecoverable from ER | **no direct** |

- Biogen-Fang (local `data/public_solubility/biogen_fang_adme/ADME_public_set_3521.csv`) cols: LOG HLM_CLint & LOG RLM_CLint (mL/min/kg), LOG MDR1-MDCK ER, LOG SOLUBILITY PH 6.8 (µg/mL), LOG PPB human/rat (%unbound). **No LogD, no Caco-2, no MLM.**
- HLM/RLM scaling ⚠️: mL/min/kg = µL/min/mg × MPPGL × LW /1000 (human MPPGL≈32 mg/g, LW≈20 g/kg → ÷0.64; rat ≈45×40 → ÷1.8). Constant offset in log space → ranking preserved; absolute calibration approximate. Cleaner HLM source = Clearance_Microsome_AZ (direct).
- TDC datasets pulled via PyTDC (`admet_group.get(...)`, scaffold split, CC-BY-4.0; downloads public data only). Added `PyTDC>=1.1.0` to requirements.txt.
- Also available (not primary): Half_Life_Obach (667, hr), VDss_Lombardo (1130), Clearance_Hepatocyte_AZ (1020, µL/min/10^6cells — different basis, not mergeable).

## Session progress log — multitask ADME (append-only, for crash/violation traceability)

Updated incrementally so a mid-session termination is traceable to the last safe step.
**Data-safety rule in force:** SMILES never read/printed. Notebook `.ipynb` files are NEVER `Read`
(their cell OUTPUTS embed SMILES + structure thumbnails — e.g. `df_all.head(3)` in cell 5 of
`Multitask_adme_preds.ipynb`); inspect only cell *sources* via a json script that strips `outputs`.
Public CSVs with a `smiles` column are passed to feature fns programmatically, never printed.

- **2026-07-07** — Prior session terminated by an API-side `violation` error (no local hook/log exists,
  so not a local guardrail). **Likely cause identified this session:** a subprocess/script error can dump
  in-house SMILES into stdout, which then crosses the wire. Concretely, a bug passed a test *DataFrame*
  where a file *path* was expected → `str(df)` (full SMILES table) went into a chemprop command and got
  echoed in the traceback. **Fix:** all chemprop subprocess stdout/stderr now redirect to LOCAL logfiles
  (`_run_quiet` in `run_chemprop_multitask.py`), never to our streams; errors raise a clean message only.
  General guardrail: any code that shells out over the in-house CSV must redirect child output to disk.
  Notebook `.ipynb` reads remain forbidden (sources-only parsing).
- ✅ Endpoint spec + `extract_adme.py` (8 endpoints → modelling space). Verified 325×8.
- ✅ In-house caches: `internal_targets.parquet`, `internal_MF.parquet` (325×4270).
- ✅ Internal-only baseline (table above).
- ✅ Deep-research public-data mapping (table above) + config `ADME_PUBLIC`/`ADME_SCALING`.
- ✅ `python/harmonize_adme.py` — per-endpoint public harmonizers to modelling space (config-driven
  `ADME_PUBLIC`/`ADME_SCALING`). Biogen path verified (rlm: 3054 cmpd, log10 µL/min/mg). TDC paths need PyTDC.
- ✅ `autoresearch/predict_adme/run_one.py` — regression experiment (modes internal/external/pooled ×
  splits random5/temporal × features MF/DS/MF+DS × model). Internal mode verified vs baseline
  (logd R²0.797, caco2 0.726/temporal 0.270, hlm 0.540); external fails gracefully w/o public cache.
- ✅ `autoresearch/predict_adme/build_public_features.py` — harmonize + attach MF/DS features to public parquets (needs PyTDC/DS).
- ✅ **Env ready (2026-07-07):** separate `chemprop` env (py3.11) created — chemprop 2.2.4, descriptastorus 2.8.0,
  PyTDC 1.1.15, torch 2.12.1, rdkit 2023.9.6. **NOT** the `ML` env (which keeps xgboost 3.2, rdkit 2025.9.3).
  Two-env split documented in README.md. build_public_features + Chemprop run in `chemprop`; XGB single-task
  arms run in `ML` (chemprop env lacks xgboost); parquet feature caches are portable between them.
  ⚠️ rdkit differs across envs (2025.9.3 vs 2023.9.6) — keep one model's features from a single env.
- ✅ **Campaign complete (2026-07-07).** Public features built (caco2 910, hlm 4189, logd 4200, ppb 1808,
  rlm 3054; MF+DS, 4474 cols). Single-task campaign = 168 experiments (`logs/autoresearch.jsonl`,
  `logs/summary.md`). Chemprop multitask temporal+random @50 epochs (`chemprop_run/metrics_*.json`). See Results below.

## Evaluation splits — how public data enters training (single-task, `run_one.py`)

Public data (**B**) only ever AUGMENTS TRAINING; the test set is ALWAYS held-out in-house (**A**). So every
mode is scored on the identical in-house labels → directly comparable. Three modes × two splits:

**random5** — `KFold(5, shuffle, seed=42)` over in-house A only:
- `internal` (baseline): fold i trains on `A\Aᵢ`, predicts held-out `Aᵢ`. (e.g. fold1 train A₂A₃A₄A₅ → predict A₁)
- `pooled`: **B fixed in every fold's train**, only A rotates — fold1 train `(A₂A₃A₄A₅ + B)` → predict A₁;
  fold2 train `(A₁A₃A₄A₅ + B)` → predict A₂; … B is never in the test fold.
- `external`: train on **B only** (no A in training) → predict held-out `Aᵢ` ("can public alone predict us?").
The 5 held-out predictions are concatenated out-of-fold; the metric is computed once over all of A.
It is a TEST fold (no per-fold hyperparameter tuning, no early stopping → no tuning leakage).

**temporal** — replaces the 5-fold with a single split: A sorted by SRB id, oldest 70% → train, newest 30% →
test; B still fixed in train (pooled/external). Tests prospective extrapolation to newer chemistry.

Leakage guard: experimental B is InChIKey-deduplicated against A, so a B twin of a test compound can't leak into train.
(Novartis stacking surrogate is trained on Novartis only, so its prediction-as-feature for A is also leakage-free.)

## Results — multitask ADME regression (2026-07-07)

**Single-task: does experimental public data beat the in-house-only baseline?** (champion RF, in-house
Pearson r² on the same test; `logs/summary.md`). Δ = best-public − baseline (both random5 and temporal):
| endpoint | rand5 baseline→best (Δ) | temporal baseline→best (Δ) | verdict |
|---|---|---|---|
| logd | 0.821→0.848 (+0.026) | 0.141→0.355 (**+0.214**) | public HELPS (pooled) |
| hlm | 0.590→0.664 (+0.074) | 0.015→0.100 (+0.085) | public HELPS |
| rlm | 0.154→0.526 (**+0.372**) | 0.068→0.217 (**+0.149**) | public HELPS (pooled) |
| ppb | 0.509→0.628 (+0.119) | 0.118→0.328 (**+0.210**) | public HELPS |
| caco2 | 0.744→0.726 (−0.018) | 0.879→0.833 (−0.046) | baseline wins (internal already strong) |
| solubility | 0.703→0.787 (+0.085) | 0.044→0.313 (**+0.269**) | public HELPS (thermo+kinetic exp) |
| mlm | — no experimental public — | — | multitask/pseudo-label only |
| mdck (MDR1) | — no experimental public — | — | multitask/pseudo-label only |

- **Under in-house Pearson r², public data still HELPS logd/hlm/rlm/ppb** — most on the temporal split
  (+0.15 to +0.21). caco2 is the exception: internal is already strong (0.74/0.88), so public slightly dilutes it.
- (Earlier sklearn-R² framing showed huge temporal "rescues"; those were calibration fixes — the correlation
  gains are the more modest, honest numbers above.) Best public config is usually **pooled** (public fixed in
  train + internal CV) — external-only ranks well but has a calibration offset (esp. caco2/rlm) that pooling
  corrects. DS (Descriptastorus) helps logd temporal; MF+DS best at random.

**Multitask Chemprop D-MPNN (8 masked endpoints).** exp-only vs all-sources (+Novartis+ADMETlab) pooled as
train; in-house Pearson r² on internal test (`run_chemprop_multitask.py <split> 50 <exp|all>`; metrics_<split>_<mode>.json).
Temporal split (reliable test coverage n=62–98):
| mode | logd | mlm* | hlm | solubility | mdck* | ppb |
|---|---|---|---|---|---|---|
| exp-only | 0.131 | 0.350 | 0.047 | **0.272** | **0.243** | **0.070** |
| all-sources | **0.430** | 0.360 | **0.109** | 0.105 | 0.132 | 0.001 |
(rlm/caco2 temporal n/a: 0 / all-constant test compounds under the global split.)
- **Random/interpolation: all-sources beats exp for nearly every endpoint** (logd 0.88→0.91, hlm 0.59→0.73,
  mlm 0.72→0.86, caco2 0.80→0.84, mdck 0.46→0.66) — but random per-endpoint n is tiny (3–33), directional only.
- **Temporal: mixed.** all-sources hugely helps **logd (0.13→0.43, best logd temporal seen)** and hlm, ties mlm,
  but HURTS solubility/mdck/ppb (pseudo-label noise dominates on newest compounds). → one giant all-8/all-sources
  model is NOT uniformly best; motivates endpoint-grouping + per-group source selection.
- ⚠️ Chemprop metrics use a single 20% (random) / newest-30% (temporal) holdout over ALL compounds, so per-endpoint
  test n is small (9–33 random) and NOT directly comparable to single-task OOF; sparse endpoints (rlm n=3, ppb n=6,
  sol n=9) are noisy. Temporal R² all negative (extrapolation hard), Spearman positive for logd (0.54)/mlm (0.23).
- ⚠️ Global temporal split leaves sparse endpoints uncovered: caco2 temporal test = 14 compounds ALL at one value
  (assay ceiling) → unscorable; rlm temporal test = 0. Single-task per-endpoint temporal split covers these better.

## Full-dataset re-run (2026-07-13, `run_full_campaign.sh`) — RF results VALID, Chemprop PENDING re-run

Re-ran the campaign with UNCAPPED public data: solubility 10K→**106K**, Novartis 6K→**273K** (full),
ADMETlab 10K→**63K** (full); config `NOVARTIS_SUBSAMPLE`/`ADMETLAB_SUBSAMPLE: full`. Experimental
TDC/Biogen sources were already full (logd/hlm/caco2/ppb/rlm) → those single-task numbers unchanged.
Capped run preserved in `logs/capped_backup/`. RF `n_jobs` capped at 32 (256 gave NO speedup —
RF tree-parallelism saturates ~32 on the 4.5k-feature matrix; ~7 min/fit on 273K, benchmarked).

**Full vs capped, TEMPORAL (in-house Pearson r²) — where full data moved the needle (RF, valid):**
| endpoint | capped best | full best | source of the gain |
|---|---|---|---|
| ppb | 0.305 | **0.496** | full Novartis+exp (source_contribution EXP+NVS); +0.19 |
| solubility | 0.313 | **0.426** | full 106K experimental (single-task external/all); +0.11 |
| mlm | 0.211 | **0.326** | full Novartis (novartis_experiments nvs_pooled); +0.115 |
| mdck | 0.414 | 0.441 | full Novartis (nvs_external); +0.03 |
| logd | 0.416 | 0.417 | ~unchanged (ADMETlab-full negligible; exp already full) |
| caco2 | 0.882 | 0.886 | internal still wins; pooling public dilutes |
| hlm / rlm | 0.031 / 0.129 | 0.018 / 0.116 | weak either way (small n / weak temporal signal) |

- **Confirms the size hypothesis at scale:** full public data most helps EXTRAPOLATION (temporal) on the
  data-hungry endpoints — **ppb, solubility, and the gap endpoint mlm**. Interpolation (random5) already
  saturated at the cap (e.g. solubility 0.703→0.778, ≈cap level) → full sets buy prospective, not interpolative, accuracy.
- **Gap endpoints (mlm, mdck) confirmed reliant on full Novartis pseudo-labels** (mlm temporal 0.21→0.33).
- Provenance: `logs/summary.md`, `logs/source_contribution.json`, `logs/novartis_experiments.json`.

**⚠️ Chemprop multitask + Chemeleon (Phase 3) results from this campaign are INVALID — stale-checkpoint bug.**
`run_chemprop_multitask.py` / `task_transfer_gain.py` selected the prediction checkpoint via
`sorted(rglob('best*.ckpt'))[0]` (LEXICOGRAPHIC), which picked a prior capped run's checkpoint
(`best-epoch=14`, Jul 7) over the full-data one (`best-epoch=24`, Jul 12) left in the same model_dir →
the "full" Chemprop metrics matched capped to 8 sig figs. **Fixed 2026-07-13** (select newest ckpt by
mtime). Phase 3 (multitask ± Chemeleon + grouping_validation, temporal+random) must be RE-RUN with the
fix; do NOT trust `chemprop_run/metrics_*` from the 2026-07-10 campaign. Capped Chemprop/Chemeleon
numbers in the 2026-07-07 results below remain valid (they were self-consistent within that run).

**Chemprop combined-CSV redesign (2026-07-13): multi-endpoint pseudo-label sources now WIDE.** Previously
`build_combined` appended each public source per-endpoint, so a Novartis compound (predicts all 7 endpoints)
became 7 single-endpoint rows → 2,351,939-row combined CSV, Novartis 81%, and the D-MPNN re-encoded each
Novartis molecule 7× per epoch. Now Novartis (273,638×7) and ADMETlab (63,136×5) are consolidated to ONE
wide row per compound (merge on the unique synthetic `compound` id — NOT smiles, which is non-unique in
ZINC/ChEMBL and explodes the join to 18M rows). Experimental sources stay per-endpoint (each is a distinct
single-endpoint dataset). Combined CSV: **2,351,939 → 457,567 rows (5.1×)**, same ~2.35M supervised label
cells, each molecule encoded once/epoch (≈5× faster, correct per-epoch weighting). Shared helper
`run_chemprop_multitask.public_rows(endpoints, mode)`; `task_transfer_gain.build` reuses it (target-restricted).

## Endpoint associations for multitask grouping (2026-07-07, data-driven)

Spearman correlation between endpoints on in-house co-measured compounds → hierarchical clustering
(cheap proxy for "which tasks belong together"; rigorous = Standley 2020 pairwise transfer-gain). Clusters:
- **Metabolic clearance {hlm, mlm, rlm}** — strong & robust: mlm–rlm 0.80, hlm–mlm 0.70, hlm–rlm 0.53 (large n).
- **Lipophilicity axis {solubility, logd}** — logd–sol −0.58 (anti-corr but linked); logd also bridges clearance
  (logd–hlm 0.42, logd–rlm 0.52).
- **Isolates: caco2, mdck, ppb** — weak/no correlation with others. **caco2–mdck ρ=0.01** (n=28): grouping the two
  "permeability" endpoints is NOT data-supported here; mdck is essentially an isolate (also weakest single-task).
- Groupings to test in multitask: {hlm,mlm,rlm} (endorsed), {sol,logd}, {caco2,mdck} (hypothesis test), all-8, singletons.

## Dataset SIZE effect — solubility 10K vs full 106K (2026-07-07, `solubility_size_experiment.py`)

Public solubility (thermo+kinetic exp), champion RF MF+DS, in-house Pearson r² on internal test:
| split | internal | pooled 10K | pooled full(106K) | external 10K | external full |
|---|---|---|---|---|---|
| random5 | 0.691 | 0.733 | 0.731 | 0.418 | 0.544 |
| temporal | 0.009 | 0.091 | **0.214** | 0.095 | **0.227** |
- **Interpolation saturates at 10K** (0.733≈0.731) but **temporal extrapolation ~doubles with the full set**
  (pooled 0.091→0.214). More diverse public data helps prospective prediction. → for the BEST deployable model,
  use FULL public sets; subsampled pseudo-labels (Novartis 6K, ADMETlab 10K) likely leave temporal gains on the table.

## Standley pairwise transfer-gain — which endpoints to multitask together (2026-07-07)

`task_transfer_gain.py`: train Chemprop on each single task + each pair, gain[A][B]=R2(A|{A,B})−R2(A|{A}),
in-house Pearson r² on internal test. Modes: internal / exp (+experimental public) / all (+Novartis+ADMETlab).
- **internal-only is UNRELIABLE** (confirmed the ~325-compound concern): matrix dominated by small-n artifacts —
  rlm (n=39) shows absurd +0.5–0.7 gains from any partner; most pairs negative from diluting tiny data. Only
  robust signal: (hlm,mlm) on temporal (the clearance pair). → not usable for grouping decisions.
- **exp / all modes** pool public data so each task has thousands of TRAIN points (decision-relevant; matches
  the deployed model). ⚠️ augmentation fixes TRAIN volume, not TEST — internal test still small for sparse
  endpoints, so trust logd/hlm/mlm/solubility (n 120–324) most. Matrices still noisy/mode-dependent; robust signals:
  - **mlm gains most from multitasking** (single 0.28→ +0.26–0.34 paired with clearance/lipophilicity in random/exp);
    strongest partners rlm/sol/logd/hlm → supports putting mlm in the clearance/lipophilicity bloc.
  - **solubility gains broadly** (random/all: +0.13–0.29 from many partners; weak single→transfer helps).
  - **logd is self-sufficient** (single ~0.88, partners ~neutral).
  - **mdck & ppb unstable/mostly-negative** → likely better ALONE (or noise-limited).
  - Conclusion: transfer-gain is directional, not decisive at this test size → settle grouping by DIRECT validation
    (below). Artifacts `task_transfer_gain_<split>_<mode>.json`.

## Endpoint-grouping validation — direct multitask test (2026-07-08, `grouping_validation.py`)

Chemprop all-sources, per-endpoint in-house Pearson r² on internal test. Candidate groups vs single-task
& all-8 references. **Temporal (deployment):**
| endpoint | single | all-8 | clearance{hlm,mlm,rlm} | metab_lipo{+logd,sol} | best grouping |
|---|---|---|---|---|---|
| hlm | 0.20 | 0.11 | **0.33** | 0.29 | **clearance group** (≈3× all-8) |
| mlm | 0.30 | 0.36 | 0.39 | **0.42** | clearance/metab group |
| logd | 0.18 | **0.43** | — | 0.23 | all-8 |
| solubility | **0.21** | 0.11 | — | 0.05 | **single-task** (grouping HURTS) |
| mdck | 0.08 | **0.13** | — | — | all-8 (perm-pair 0.12 = no better) |
- **Focused clearance multitask {hlm,mlm,rlm} is the standout** — hlm 0.11→0.33, mlm best-or-tied — a small
  biologically-coherent group beats the giant all-8. **solubility is best ALONE**; **logd best in all-8**;
  **{caco2,mdck} permeability pairing does NOT beat all-8** (confirms corr ρ≈0 — mdck wants broad transfer, not caco2).
- Random/interpolation: all-8 ≈ best (more tasks/data help interpolation); grouping matters mainly at temporal.
- **Design implication:** no single architecture wins all endpoints → deploy a **clearance-specific {hlm,mlm,rlm}
  model**, **solubility single-task**, and logd/mdck/caco2/ppb from all-8 or single-task per the tables.
- Artifacts: `grouping_validation_<split>.json`.

## Source-contribution factorial — which dataset helps most, per endpoint (2026-07-07)

`source_contribution.py`: per endpoint × split, champion RF on MF+DS, pools internal + every non-empty
subset of the available sources {EXP experimental, NVS Novartis, ADM ADMETlab}, in-house Pearson r² on
the same internal test. Winner per cell (temporal = deployment-relevant):
| endpoint | internal | best source(s) | r² (Δ) |
|---|---|---|---|
| logd | 0.22 | **EXP+ADM** (combo) | 0.42 (+0.20) |
| ppb | 0.06 | **ADM** | 0.31 (+0.25) |
| mlm* | 0.08 | **NVS** | 0.23 (+0.15) |
| solubility | 0.01 | **EXP** | 0.12 (+0.11) |
| mdck* | 0.30 | **NVS** | 0.40 (+0.10) |
| rlm | 0.07 | **NVS** | 0.13 (+0.06) |
| hlm | 0.00 | NVS | 0.05 (+0.05) |
| caco2 | **0.88** | none (internal best) | — |

- **No single best dataset — sources specialize:** NVS → gap endpoints mlm/mdck (+rlm/hlm); ADM → ppb & (with
  EXP) logd; EXP → solubility. caco2 needs nothing (internal already 0.88).
- **More data ≠ better.** Pooling all three usually DILUTES: caco2 EXP+NVS+ADM 0.60 vs internal 0.88;
  logd EXP+NVS+ADM 0.33 vs EXP+ADM 0.42. Best = one well-chosen source or a specific pair.
- **Interpolation (random5):** internal wins/ties almost everywhere (only solubility+EXP, ppb+NVS, rlm+EXP+NVS
  help) — in-house interpolates its own chemistry; public/pseudo-labels pay off at temporal extrapolation.
- **Gap endpoints rescued by pseudo-labels** (their only signal): mdck 0.30→0.40, mlm 0.08→0.23 (Novartis).
- Artifact: `logs/source_contribution.json` (r2 + spearman per config).

## Novartis/NIBR predictions — 3-approach study (2026-07-07)

`data/protacdb2.0_zinc_chembl_dataset.csv` = Novartis/NIBR **in-silico ADME predictions** (not experimental)
on 273,706 public compounds (ZINC 200k, ChEMBL 70k, PROTAC-DB 3269). Columns `pred(...)` incl. the giveaway
`pred(Direct NIBR LogD7.4)`. Covers our two GAP endpoints (mlm via `pred(mLM LogCLint)`, mdck via
`pred(LE-MDCKv2_LogPapp)` — LE-MDCK proxy, not MDR1). Config `NOVARTIS_PUBLIC`; subsampled to 6000
(PROTAC-DB fully + ChEMBL/ZINC) with MF+DS features → `public_novartis_<ep>.parquet`. Modules:
`build_novartis_features.py`, `novartis_benchmark.py`, `novartis_experiments.py`.

**A1 — Benchmark (Novartis pred vs EXPERIMENTAL public, InChIKey overlap):** pseudo-labels RANK experimental
values well; Spearman / Pearson-r² / r2_det(calib), n: ppb 0.80/0.67/0.66 (168) · caco2 0.74/0.60/0.52 (100) ·
rlm 0.73/0.59/0.40 (41) · hlm 0.70/0.46/**−0.63** (171) · logd 0.69/0.51/0.43 (369). hlm ranks/correlates well
but absolute scale biased (r2_det<0). caco2 r2_det>0 confirms the identity unit-map (NIBR Caco-2 LogPapp is
log10(1e-6 cm/s), no offset — unlike Wang).

**A2/A3 — using the pseudo-labels (champion RF, MF+DS, same internal test as campaign).** Configs: `nvs_only`
(train ALL Novartis → predict every in-house cmpd, pure transfer), `nvs_pooled`/`expnvs_pooled` (distillation),
`stack`/`stack_exp` (Novartis-surrogate prediction as an extra feature). **R² = in-house Pearson r².**
Temporal split (deployment-relevant), R² per config:
| endpoint | internal | best Novartis config | its R² | verdict |
|---|---|---|---|---|
| mdck* | 0.30 | nvs_external / nvs_pooled | **0.41 / 0.40** | **Novartis win** (gap endpoint) |
| mlm*  | 0.08 | nvs_pooled (distill)      | **0.21** | **Novartis win** (gap endpoint) |
| ppb   | 0.06 | stack_exp                 | **0.28** | **win** (Novartis+exp) |
| logd  | 0.22 | stack                     | **0.30** | win (stacking) |
| caco2 | 0.88 | stack                     | 0.89 | tie (already high; pooling HURTS → 0.70) |
| rlm   | 0.07 | nvs_pooled                | 0.13 | marginal |
| hlm   | 0.00 | (all ≈0.03)               | 0.03 | none (weak temporal signal) |

- **Re-read under Pearson r², the picture inverts vs the earlier sklearn-R² report.** The old "big wins"
  (rlm, caco2) were *calibration* fixes: caco2 correlation is already 0.88 internally, and raw pooling actually
  *lowers* it (0.70). The GENUINE correlation gains from Novartis are on the **gap endpoints mdck (0.30→0.41)
  and mlm (0.08→0.21)** — where it's the only signal — plus **ppb (0.06→0.28, stack_exp)** and **logd (0.22→0.30, stack)**.
- **Stacking (A2)** is the safe default: at random5 stack ≈ internal everywhere (never hurts); at temporal it wins
  logd and ties caco2. **Distillation (A3)** is what helps the gap endpoints (mdck/mlm) and ppb, but can lower
  caco2. **Pure transfer (`nvs_only`)** correlates decently for logd 0.66 / caco2 0.44 / mdck 0.42 / mlm 0.26
  (Pearson r²), poorly for rlm/ppb — usable for ranking gap endpoints, not for absolute values.
- **Bottom line:** use Novartis mainly for the **gap endpoints (mlm, mdck)** via distillation/pooling, and as a
  **stacking feature** (stack/stack_exp) elsewhere where it's low-risk. It does NOT improve caco2 (internal already strong).
- Artifacts (both carry `r2` Pearson + `r2_det` sklearn): `logs/novartis_benchmark.json`, `logs/novartis_experiments.json`.

**Decisions (both RESOLVED):** (1) mdck **restricted to MDR1-MDCK** (2026-07-07) via config
`ADME_ENDPOINTS.mdck.filter {col: mdck_Cell line, contains: MDR1}` — drops 2 MDR2/MDR3 compounds
(n 109→107). Cell-line labels: `MDR1-MDCKⅡ` (106, unicode Ⅱ) + `MDR1-MDCK II` (3). PgP-inhibitor columns
excluded. Cleaner single-assay signal slightly *improved* the baseline: RF random5 0.421→**0.479**, temporal
0.254→**0.288**. `extract_adme` now supports an optional per-endpoint `filter` (contains-match, nulls the rest).
(2) solubility `…Thermodynamic Solubility (1)` — **RESOLVED**:
confirmed identical to the column used in `Solubility_sticky_cmps.ipynb` (cell 6: `Thermodynamic Solubility:
Thermodynamic Solubility (1) (μM)`), so consistent with prior solubility work. That experiment was classification
(label ≥ CUTOFF_SOL 10 µM); this ADME work models the same column as regression (log10 µM).

## Public solubility datasets (2026-06-25 deep-research + download)

Downloaded to `data/public_solubility/` (23 MB). Provenance: deep-research workflow `wf_842b6508-728`.

| Folder | Set | Rows | Measures | Units | License | Merge tier |
|---|---|---|---|---|---|---|
| `aqsoldbc_solcuration/clean,cure/` | SolCuration 7-set (aqsol, aqua, chembl, esol, ochem, phys = thermo; kinect = kinetic) | clean+cure each | 6 thermo + 1 kinetic | **logS** (unified `smiles,logS,weight`) | CC-BY 4.0 | **Core mergeable** (exclude kinect) |
| `aqsoldb/` | AqSolDB curated | 9,982 | aqueous (mixed types) | logS (mol/L) | CC0 1.0 | Pretrain (curate) |
| `llompart_curated/` | AqSolDBc (8,047) + OChemCurated (7,463) | — | thermo, quality-assessed | logS | Etalab/CC-BY | **Core mergeable** |
| `pharmabench/` | PharmaBench water-sol (final 11,701; preproc 14,818) | 11,701 | water, equilibrium-filtered (pH 7.0–7.6, HPLC) | **log10 nM** | open (Nature data) | **Core mergeable** (unit convert) |
| `esol_delaney/` | ESOL/Delaney benchmark | 1,128 | water solubility | logS (mol/L) | open (DeepChem) | Subset of above |
| `biogen_fang_adme/` | Biogen Fang 2023 (Polaris source) | 3,521 | aqueous @ pH 6.8 + 5 other ADME | µg/mL (log) | open | Multi-task |

**Already local (not re-downloaded):** PROTAC-PatentDB = `data/PROTAC_Patent_Compounds.xlsx` (63,136 PROTACs, mean MW 920 Da, **predicted props only, no experimental solubility**, CC-BY-NC-ND).

**Wiki-pS0 bRo5 anchor — partially recovered (2026-06-25):** the paper (Avdeef 2020, ADMET DMPK, DOI 10.5599/admet.794, PMC8915605) gives **no SMILES** (Table 1 = name + logS0 + MW for 31 bRo5 cmpds; full 3,065-cmpd DB is unreleased/book-only). Recovered **28/31** by resolving names→SMILES via PubChem (3 paper-specific paclitaxel analogs have no public structure). MW agreement 28/28 within 5%. Saved `data/public_solubility/wiki_ps0_bigmol/wiki_ps0_31_bigmol.csv`; wired into harmonizer as origin `Wiki-pS0`. NOTE: logS0 is **intrinsic** solubility (neutral form) — differs from target's apparent thermo solubility at fixed pH; close for these mostly-nonionizable big molecules. Resolver script: `scratchpad/wiki_ps0_resolve.py`.

**Solubility Challenge 2 — added (2026-06-25):** user supplied SI `data/public_solubility/solubility_challenge/ci0c00701_si_001.xlsx` (Llinàs 2020, JCIM, DOI 10.1021/acs.jcim.0c00701). Gold data in sheet `SET1 and SET2` (SET1=100 CheqSol train, SET2=32 shake-flask test); names + logS0, **no SMILES**. Recovered **129/132** (SET1 97, SET2 32/32) via PubChem. Saved `solchallenge_gold.csv`; wired into harmonizer as origins `SolChallenge-SET1`/`SolChallenge-SET2` (keep SET2 as clean hold-out). intrinsic S0 (same caveat as Wiki-pS0). Resolver: `scratchpad/solchallenge_resolve.py`.

**PROTAC-patent files moved** (2026-06-25) to `data/public_solubility/protac_patent/` (Compounds + ADMET_Overview + Patent_information). Harmonizer `PROTAC-PatentDB` source path updated accordingly.

**Manual-download remainder** (not auto-fetchable):
- **Wiki-pS0** full DB (3,065 cmpds, book-only); **PROTAC-DB 3.0** (cadd.zju.edu.cn/protacdb, structures only, no solubility, SPA).

## OpenADMET / Polaris ADMET challenge — takeaways (2026-06-25 deep research, run wf_d9b09bc4)

ASAP×Polaris×OpenADMET Antiviral (2025) + OpenADMET-ExpansionRx (2026, 370+ teams). Endpoints: KSOL, LogD, HLM, MLM, MDR1-MDCKII, potency. Winner: **Inductive Bio** (Beacon; data-centric, not novel architecture).

- **#1 lever = external/task-specific ADMET data**, far above architecture. 8/10 top teams added ADMET data; 4/5 top used proprietary. Best pretrained DL model *without* extra data had 23% higher error than winner. Data eng/augmentation > HPO ("HPO took a back seat").
- **Architecture secondary, no single winner:** XGBoost (55 descriptors) best on 4/5 ADME endpoints (SystemsCBLab); GNNs dominated the ExpansionRx top (Chemprop, multitask); Simulations Plus 3rd with descriptor-first **TabPFN/CatBoost** (TabPFN −44% MAE vs CatBoost, no HPO). → our XGBoost is well-justified; benchmark TabPFN.
- **Foundation models inconsistent** except tabular (TabPFN). Non-task-specific pretraining = mixed (MolMCL 5th, MolE 10th).
- **Validation was a differentiator:** temporal + difficulty-based holdouts, not tuning. Multitask learning helped.
- ⭐ **Per-endpoint difficulty:** LogD + protein binding easiest (R² .92–.98); clearance (HLM/MLM) hardest (R² .36–.54); **KSOL solubility = decent RMSE but POOR ranking — models act as binary soluble/insoluble classifiers, can't resolve within the soluble cluster.** → for our solubility model, **evaluate with Spearman/Kendall ranking, not just RMSE**; hold out SolChallenge-SET2.

## Solubility classifier autoresearch (2026-06-25, `autoresearch/predict_solubility/`)

Task: classify soluble (≥10 µM) vs insoluble on the 121 unique in-house thermo-solubility
compounds; max ROC-AUC. Modes: test_only (baseline CV, no public), external (public→in-house),
pooled (public + 4/5 in-house). 109 experiments, dataset-selection-first, then cut short by user.

**Verdict depends on the evaluation — interpolation vs extrapolation:**
- **Random 5-fold (interpolation):** in-house-only ≈ **0.893–0.925** ROC-AUC; public-augmented ≈ 0.906–0.927 → lift ≈ 0 (public data unnecessary within the series; random splits leak series self-similarity).
- **Temporal / prospective (extrapolation, SRB-ordered, predict newer):** public data **clearly helps**. Rolling expanding-window (73 newer cmpd, 13 pos, champion HistGB lr0.05/md6/balanced): internal-only ROC **0.745** / PR 0.38 → public+internal ROC **0.874** / PR **0.68** (+0.13 ROC, +0.30 PR). Single 30% split (4 pos) directionally agrees (0.55→0.73).
- Best public subset = `thermodynamic_experimental + kinetic_experimental` (115,848 cmpd); predicted (PROTAC-PatentDB) **hurts** (−0.04); more public data dilutes the random-CV signal.
- Tuned champion `min_samples_leaf=100` **degenerates** on the ~85–121-cmpd internal-only training (can't form a leaf → ROC ~0.5); use size-robust params for internal-only.
- At 0.5 decision threshold MCC≈0 under prevalence drift despite ROC 0.87 → **deploy with a tuned/lower threshold**; PR-AUC is the honest metric.
- **Deployable recommendation: public-augmented (pooled, thermo+kinetic, HistGB) for predicting NEW chemistry; in-house-only suffices only for within-series interpolation.**
- Notebook cells `sol_eval_*` (random5 + temporal, internal vs public+internal). Scripts: `temporal_split_eval.py`, `rolling_temporal_eval.py`. Campaign: `logs/summary.md`, `logs/dataset_selection.md`. Not reached: cutoff/feature/ensemble sweep, Chemprop (not installed).

**Regression confirms the same story (2026-06-26, RF n_est200/max_depth20/max_feat0.3/msl2/mss4; target = log10 solubility µM; 4 ID-set scenarios, in-house n=120, public thermo+kinetic_exp n=109,287, bounded 0<sol≤1e6):**
- **Random 5-fold (interpolation):** internal-only R²**0.710** / RMSE**0.643** / Spearman**0.740**; public+internal R²0.638 / RMSE0.718 / Spearman0.752 → public **does not help, slightly hurts** R²/RMSE (series dilution), ranking a wash. In-house alone interpolates fine.
- **Temporal (extrapolation, newest 30%, n=36):** internal-only R²**−2.598** / RMSE0.795 / Spearman0.326 → public+internal R²**−0.750** / RMSE**0.555** / Spearman**0.401**. Public **clearly helps** (RMSE −0.24 log, Spearman +0.07, R² +1.85) — same direction/magnitude as the classifier lift.
- ⚠ Temporal R² stays **negative even with public** (−0.75): can't beat the mean on variance-explained for genuinely new chemistry, but **ranks** (Spearman 0.40) at RMSE ~0.55 log ≈ interlab noise floor (~0.6) → useful for triage/ranking, not precise log-S. **Trust Spearman, not R²/RMSE** (matches OpenADMET KSOL takeaway). n=36 → directional.
- Script: `scratchpad/reg_eval.py` (lean float32, positional folds; no InChIKey leakage mask, fine for bRo5-vs-small-molecule). Uses the `ML_reg.K_fold_by_defined_IDs` ID-set idiom (run lean to avoid OOM on the 109k block).

## Key findings

- **Thermo vs kinetic is the mergeability rule** (Llompart 2024, s41597-024-03105-6): three thermo measures (water, apparent@pH, intrinsic S0) vs kinetic (DMSO→PBS precipitation). Don't mix assay types. AqSolDB & OChem are *undefined mixtures* needing curation.
- **bRo5 transfer is hard:** small-molecule models predict bRo5 solubility at only R²≈0.42, RMSE≈1.06 log. Public experimental bRo5 solubility is scarce (~31 molecules ≥800 Da total) → in-house fine-tuning is the only viable route.
- **Noise floor:** interlab solubility reproducibility ~0.6 log S — bounds achievable RMSE.
- **Training-size estimate:** ~15k–45k unique small molecules after dedup (depends on curation strictness); only ~31 public large molecules. Exact unique count needs InChIKey dedup (not yet run).

## Harmonizer — `python/harmonize_sol.py` (2026-06-25)

`harmonize_solubility(data_dir='data', include=, exclude=, collapse_per_type=, add_inchikey=, out_csv=)`
→ tidy long table `[smiles, origin, type, solubility(µM)]`. Converts every source to µM to match the
in-house target. Uses fresh ChEMBL API pull (not the stale SolCuration ChEMBL 26) + SolCuration `cure/`
for AQUA/PHYS/ESOL/OChem/AqSolDB/KINECT + PharmaBench + Biogen-Fang + PROTAC-PatentDB (ADMETlab predicted).
Conversions: logS→`10**logS*1e6`; log10(nM)→`10**v/1e3`; nM→`v/1e3`; µg/mL→`v*1e3/MW`; log10(µg/mL)→`10**v*1e3/MW` (MW via RDKit).

**Output (2026-06-25):** 237,702 labelled rows, 10 origins. Types: kinetic_exp 93,851 · thermo_predicted 63,136 ·
mixed 57,936 · **thermo_experimental 22,779**. `collapse_per_type=True` → 214,107 unique (InChIKey,origin,type).

**Decisions:** ProtacDB2.0 **dropped** (no solubility column, only predicted clearance/perm/CYP).
Predicted rows (PROTAC-PatentDB) kept in the same table, `type='thermodynamic_predicted'` — filter via `type`.
**Open caveat:** ~413 rows > 1e7 µM (>10 M, physically impossible source values) — add an outlier bound before training.

## Conventions / decisions

- Censored solubility (`< X`, `> X`) → float via `.str.extract(r'(\d+\.?\d*)').astype(float)` (in `Main.ipynb`).
  Harmonizer keeps censored values at their reported bound (same convention).
- Datasets organized one-folder-per-source under `data/public_solubility/`.
- Conda env for this work: `ML` (`~/miniconda3/envs/ML/bin/python`, pandas 3.0.3, rdkit 2025.09.3).
- **Property-prediction eval idiom (2026-06-25):** for any property model, evaluate with explicit train/test
  ID sets via `ML_Class.K_fold_by_defined_IDs_Classification(ML_data, ID='compound', ID_sets=[[train, test], ...],
  model=champion, col_to_rm=['compound','label','smiles'], v=True, ctf=0.5)` (regression: `ML_reg.K_fold_by_defined_IDs`).
  Build one `ML_data = concat([public, internal])` with H236 features (`rdkit_tools.compute_H236_features`), binarize
  `label` locally per cell, then express each scenario purely as `ID_sets` — keeps train/test membership explicit and
  auditable (vs. random shuffling). Patterns: **random 5-fold** = `KFold` over in-house IDs, public IDs pinned into
  every fold's train; **temporal** = sort in-house by SRB digits (`sort_values('compound', key=...)`, bigger=newer),
  oldest 70%→train / newest 30%→test, public pinned into train. Same held-out test across with/without-public arms so
  the public-data lift is directly measurable. Notebook cells `sol_eval_1a/1b/2a/2b` are the reference implementation.
  Caveat: no InChIKey leakage mask between public and internal (fine for bRo5 in-house vs small-molecule public).

## Prediction webapp — LiveDesign-style drag-drop scorer (2026-08-24, `webapp/`, v1 built)

Local-only FastAPI webapp in `ADME_ML/webapp/` (mirrors `../CDD_Vault_API/webapp` skeleton). Drag-drop an
SDF or CSV of compounds + SMILES; it scores every compound with the 8 champion RF models in the
MLTrail vault (`/data2/MLTrail_vault`) and shows a LiveDesign-style grid. (Started on **H236**; since
2026-08-26 the app prefers **H237 + `conf_recal`** per endpoint — see the H237 note below.)

**Files:** `webapp/app.py` (FastAPI backend), `webapp/static/{index.html,app.js}` (drop zone + grid),
`webapp/requirements.txt`. **Run:** `conda run -n ML python webapp/app.py` -> http://127.0.0.1:8050.
The `ML` env has mltrail/rdkit/sklearn/pandas; **fastapi+uvicorn are installed** (verified 2026-08-31:
fastapi 0.141.1, uvicorn 0.52.4). Core predict path verified on public SMILES
(2026-08-24): all 8 champions load, H236 = 4269 features, per-tree std -> confidence, inverse transforms sane.

- **Model backend:** MLTrail. Champions = registry entries with `experiment_name` starting `adme_`,
  `features_type='H236'`, `framework='sklearn'`, `model_type='single_task_regression'`.
  `registry.predict(mid, df, smiles_column='smiles', compound_id='compound')` returns modelling-space
  predictions; SDF/CSV both handled by MLTrail's `read_dataset` (H236 featurizer is RDKit-only, offline).
- **Efficiency:** featurize H236 **once**, run all 8 models on the shared matrix (avoids 8× re-featurize).
- **Predictions are in modelling space** → webapp inverse-transforms with `_TF` (ADME_build_ML.py) to raw units.
- **Grid columns:** structure (RDKit SVG, server-side) · compound_id · score (user-defined, wired later) ·
  **MPO** (client-side weighted formula, see below) · one column per endpoint. **Each endpoint cell is
  diagonally split:** bottom triangle = predicted raw value colored by cutoff (favorable/unfavorable); top
  triangle = RF confidence, diverging gradient centered at 0.5 (`SERAC_C.azure #0EA5CE` for >0.5,
  `SERAC_C.ember #E65D32` for ≤0.5).
- **Sortable columns + MPO formula (2026-08-26, app.js v2, `?v=2026-08-26`).** All work **client-side** (rows kept
  in `ROWS`; no re-scoring, no data leaves the browser). Click any header to sort (numeric high-first, blanks last;
  toggle direction). The **MPO** column reads an editable formula field that recomputes live per keystroke and
  re-sorts if MPO is the active column. Formula scope: a bare endpoint name (e.g. `logd`) = raw predicted value;
  `d('logd'[, slope])` = **desirability** in [0,1] = a sigmoid at the triage cutoff in the model transform space
  toward the favorable side (default slope 2, d=0.5 at the cutoff); `c('logd')` = confidence; helpers
  `sigmoid('logd'[, center, slope])` = desirability, center defaults to that endpoint cutoff (2nd arg overrides it in raw
  units), favorable inequality sets the direction so every term is 1=good/0=bad (== `d`); `sigmoid(x,center,slope)` =
  manual numeric form; `mean(...)` `clamp min max exp log abs pow`. Default formula shows cutoffs explicitly
  (`mean(sigmoid('solubility',10), sigmoid('logd',3), ...)`). Eval = `new Function` + `with(scope)`,
  blocked tokens (`=>`, `function`, backtick, `window/document/fetch/this`) — safe enough for a localhost single
  user. Default formula = equal-weight `mean(d(ep)...)` over all endpoints. `/api/models` now also returns each
  endpoint `transform` (needed to build `d()`). Logic tested in `scratchpad/test_mpo.js` (14 checks: cutoff→0.5 in
  each transform space, favorable direction, `sigmoid(logd,3)` literal, weights, invalid-row→null, token guard, sort).
- **Hover preview + per-row CSV selection (2026-08-27, `?v=2026-08-27b`).** Hovering a structure cell shows a
  floating high-resolution preview. It is a FRESH RDKit render (`svg_hi`, 400×300, absolute `bondLineWidth=1.2`,
  `scaleBondWidth=False`) carried per row — NOT the 150×100 thumbnail scaled up (that magnified strokes ~4.8px;
  the fresh render is ~1.1px, thin + clean). `_svg` gained a `bond_line_width` arg.
  Each row has a checkbox (+ a header select-all, with indeterminate state) that flags it for download; `r.selected`
  lives in `ROWS` and survives sort/re-render. **Download is now client-side**: `toCSV(selected rows)` builds the CSV
  in-browser (columns compound, smiles, mpo, score, then `<ep>_pred`/`<ep>_confidence`) and saves via a Blob — so it
  respects the checkboxes and adds MPO+Score. The server `/api/download` endpoint is now unused (left as-is).
- **Collapsible panels + row filters (2026-08-31, `?v=2026-08-31`).** The MPO block is now a `<details class="panel">`
  and is **collapsed by default** (its `✓ applied` / `✗ error` status moved into the `<summary>`, so a bad formula is
  visible while closed). A second panel, **Filters**, holds a LiveDesign-style term builder: one row per term =
  `property` · `operator` (`> >= < <= = ≠`) · `value`, with "⊕ Add a term" / "⊖ Remove term" and an all-vs-any
  (AND/OR) selector. Filterable properties = every endpoint (raw value), every `<ep>_conf` (RF confidence),
  plus `MPO` and the manual `Score`. Everything is client-side: `activeTerms()` drops incomplete terms,
  `passRow()` tests a row, and `renderRows()` now renders `VISIBLE` (= ROWS that pass) while keeping the ROWS index
  in each cell's `data-i`, so sort, the Score boxes and the hover preview stay correct. A missing/unparsed value
  fails a term. Selection follows the filter: select-all and the CSV download act on the VISIBLE rows only.
  Filtering never re-scores — it only hides. Logic tested DOM-free by slicing the filter section out of `app.js`
  (`scratchpad/test_filters.js`, 15 checks on synthetic rows: AND/OR, `_conf` vs value, all 6 operators,
  incomplete terms inert, null-value rows excluded, mpo/score fields, negative thresholds, field list).
  Static files are read per request, so a running server picks the change up without a restart.
- **Column show/hide, an extra model column, and a color menu (2026-08-31, `?v=2026-08-31b`).** Three additions:
  (1) **Columns panel** — a checkbox per column (Structure / Compound / Score / MPO / every model column) with
  "show all" and "hide all endpoints". `HIDDEN` is a key set; `renderHeader()` (which replaced the old in-place
  arrow patcher `refreshHeader`) and `renderRows()` skip a hidden key. Hiding is display-only: the column is still
  scored, still filterable, and **still written to the CSV** (`toCSV` iterates `MODELS.columns`, never `HIDDEN`).
  (2) **Extra (non-ADME) model column** — new config list `webapp.extra_models`, loaded by `_load_extras()`.
  First entry = **MLTrail id 19 `Px_activity_1_12_rf_H237`** (experiment_measure `proteomics_activity`,
  `single_task_classification`, H237, roc_auc 0.645 / pr_auc 0.761, n_train 4859, classes 3271/1588, positive =
  `1 <= ndown <= 12`, i.e. the single/low-activity class). The column shows **P(positive)** with `favorable` at a
  config `threshold` (0.5), and its **confidence = the decision margin |2p−1|** — a per-tree std is useless for a
  binary forest (trees vote 0/1, so std ≈ sqrt(p(1−p)) adds nothing to p). It reuses the H237 matrix already
  featurized for the ADME champions, so it costs one extra `predict_proba`. It is **excluded from the MPO**:
  the MPO scope is built from `MODELS.endpoints`, while the grid/filters use `MODELS.columns` = endpoints + extras
  (`_byKey` vs `_byCol`). It is filterable (`px_activity`, `px_activity_conf`) and exported to the CSV.
  (3) **Colors panel** — four pickers (favorable value, unfavorable value, confidence above/below the split) over a
  live `PAL` copy, with a reset to the SERAC config defaults (olive/azure/ember; the MPO column follows the
  favorable color, and the legend swatches track the pickers). The config hex is upper-case, so it is lower-cased
  at load — `<input type=color>` rejects `#0EA5CE`. The choice lasts for the page load only.
- **Value color is a diverging FADE, not a switch (2026-08-31, `?v=2026-08-31c`).** The lower-left triangle used a
  boolean `favorable` flag, so 9.9 and 10.1 µM looked opposite. New `gradPos(col, v)` puts the value on a 0–1
  favorable axis = sigmoid of the distance to the triage cutoff **in modelling space** (log units for a log
  endpoint), with the favorable inequality setting the direction — the same desirability the MPO uses. `predColor(t)`
  then fades: palest at the cutoff (alpha 0.10) and saturating to alpha 0.90 far from it, olive above / red below.
  Steepness = the column's `color_slope` × a `value fade` slider in the Colors panel (0.25–4, default 1).
  `color_slope` is config: `webapp.color_slope: 2.0` for the endpoints (one log unit past the cutoff ≈ 88% of the
  color) and **8.0 for the classification column**, whose probability moves at most 0.5 from its threshold. The
  legend's two hard swatches became one gradient bar. Tests: `scratchpad/test_colors.js` (19 checks: 0.5 exactly at
  the cutoff, palest fill there, monotone with no jump, direction flip for lower-is-better endpoints, alpha
  saturation, the sharpness knob, the steeper classification slope, grey for null/uncomputable values).
  Tests: `scratchpad/test_filters.js` now 20 checks (adds the extra column in the filter fields, MPO-scope
  exclusion, and CSV-keeps-hidden-columns), plus a live end-to-end run on **public SMILES only** (ethanol/benzene/
  aspirin/caffeine + a bad SMILES) against a throwaway instance on port 8051: 9 prediction columns, unparsed row
  null. **CAUTION: a running server must be restarted to pick up `app.py`** — static files reload on their own.
- **Confidence:** per-tree std across `rf.estimators_` → `ML_Reg.uq_std_to_confidence`. **sigma (v1) =
  per-endpoint training-label std**, taken from the champion's archived training set
  (`registry.load_training_set(mid)['label'].std()`, modelling space — consistent with the std units).
- **CUTOFFS (raw units) + signs** (favorable when): solubility 10 (≥), logd 3 (<), caco2 10 (≥), mdck 10 (≥),
  ppb 1 (<, % unbound), hlm/mlm/rlm 12 (<, µL/min/mg — WEAK: measurement-floor convention, no verified triage cutoff).
- **Privacy:** binds `127.0.0.1` ONLY. Predictions/SMILES/ids render in the LOCAL browser (same as the CDD
  app's `/api/summary`); nothing crosses to any cloud. Claude must never Read/echo the prediction outputs.
- **Confidence calibration — 4-way comparison implemented (2026-08-25).** `ML_Reg.calibrate_confidence_params(cv_pred_df)`
  learns per-endpoint params from the **augmented_cv** arm (has tree-std + true residual): `rmse_cv`, `label_std`,
  a nonneg linear fit `|resid| ~ a + b*std` (recalibrates the under-dispersed RF tree-std into error units), and
  the sorted split-conformal nonconformity scores `|resid|/std`. `ML_Reg.apply_confidences(pred_df, calib)` then
  adds 4 comparison columns to every arm's pred_df: **conf_labelstd** (v1, `exp(-std/label_std)`), **conf_rmse**
  (`exp(-std/rmse_cv)`), **conf_recal** (`exp(-clip(a+b*std,0)/rmse_cv)`), **conf_conformal**
  (`frac(nonconf <= rmse_cv/std)` = split-conformal P(|err|<=RMSE_cv)). `OUTPUT.assess_predictions` computes the
  calibration once (from augmented_cv) and applies it to all 6 arms; params stored under
  `metrics_results[k]['_calibration']` for the webapp. `OUTPUT.predict_and_record` no longer writes the old inline
  `confidence`. Tests: `tests/test_confidence.py`. Pick a winner later, then wire that one into the webapp
  (replacing the v1 training-label-std path). NOTE: cached pickles from the earlier run predate these columns —
  delete `output/results/20260825_metrics/*.pkl` to recompute with the 4 conf_* columns.
- **Leakage fix — leave-one-fold-out calibration (2026-08-25).** The v1 above calibrated AND scored the same
  `augmented_cv` residuals (in-sample) — biases the variant comparison toward the flexible calibrations. Fixed:
  `K_fold_by_defined_IDs` now returns a **`fold`** column (1-indexed test fold; single split -> all 1).
  `ML_Reg.apply_confidences_lofo(pred_df)` does **cross-conformal / CV+**: each fold scored by a calibration fit
  on the OTHER folds (no extra model fits; needs >=2 folds, else None). `OUTPUT.assess_predictions`: **CV arms**
  (internal_cv, augmented_cv) use LOFO; **single-block arms** (ext_->_internal, both temporals, ext_->_temp)
  calibrate on `augmented_cv` EXCLUDING that arm's test compounds (fully honest), falling back to the pooled
  deploy calibration when <30 rows remain (notably `ext_->_internal`, whose test = ALL internal, so nothing is
  left — augmented_cv OOF is internal-only because public is train-only in fold_ids_aug). Pooled augmented_cv
  stays as `_calibration` for the webapp (no leakage at deploy: new compounds are unseen). conformal kept as a
  free 4th column under the same scheme (user deprioritized it). Tests cover LOFO order/bounds/monotonicity + the
  <2-fold None path.
- **InChIKey-grouped CV folds — anti-leakage (2026-08-26, `DATA._grouped_folds`, config `FOLD_GROUP_BY_INCHIKEY: true`).**
  The confidence-vs-residual plot looked "too clean" (all 4 variants Spearman ≈ −0.71). **That equality is a math
  identity, not leakage:** every conf variant is a strictly monotone transform of one number (`uq_std`), and Spearman
  is invariant under monotone transforms, so all four = `−Spearman(uq_std, |resid|)`; calibration only rescales the
  x-axis. Code audit of `ML_Reg.K_fold_by_defined_IDs` confirms `pred_y`/`uq_std`/`real_y` are genuine OOF (train/test
  ID-disjoint; public added to TRAIN only; public InChIKey-twins of internal already dropped in
  `get_internal_public_sets`). The one real exposure: `_endpoint_ML` dedups by SMILES only, so internal InChIKey twins
  (same molecule, different SMILES) could split across random KFold folds and inflate the signal. Fix:
  `select_best_combo_and_update` now builds the 5 folds **grouped by InChIKey** (unique groups KFold-split with
  seed 42, each compound inherits its group's fold; missing `_ik` -> singleton) so twins never straddle a split.
  Deterministic; a no-op when there are no twins. Toggle off with `FOLD_GROUP_BY_INCHIKEY: false`. Test:
  `tests/test_build_ml_data.py::test_grouped_folds_no_inchikey_twins_across_folds`. **NOTE:** this changes CV +
  conf_recal calibration numbers, so cached metrics pickles must be deleted to recompute, and models must be
  **re-deployed** for the new calibration to reach MLTrail/the webapp.

## NVS negative-transfer subset search (2026-08-26, `python/nvs_subset_search.py`, mdck pilot — results pending)

**Motivation.** The grouped-fold `augmented_cv` R² plot (all 8 endpoints, transfer/internal-CV/augmented-CV) shows
augmentation HELPS logd/ppb/rlm/solubility but HURTS the NVS-only endpoints (mdck/hlm/mlm/caco2): augmented < internal.
Consistent with the 2026-07 cleaning sweep (drop predicted NVS/ADM for mdck/mlm/caco2) — NVS = Novartis-NIBR
**predicted** labels with a **systematic upward bias** (mdck: wrong assay variant, +1.6–1.9). New question: is there a
predictive SUBSET of NVS that beats internal-only? `NVSSubsetSearch(data, output, params, endpoint='mdck')` runs in the
user's kernel on a built DATA/OUTPUT and tests four levers vs the internal-only + all-NVS baselines:
- **S1 distance** (`distance_curve`): augment with NVS within a swept Tanimoto distance of the internal TRAIN fold
  (leakage-free — distance to train rows only, per fold; quantile grid). Expect an inverted-U if a near subset helps.
- **S2/S4 scaffold** (`scaffold_greedy`): cluster NVS by Bemis-Murcko GENERIC scaffold (top-K frequent + 'other'),
  forward/backward greedy group selection maximizing augmented CV R².
- **S5 bias-correction** (`biascorrect_curve`): per train fold, affine-calibrate NVS labels toward internal on
  near-neighbour pairs (max-sim ≥ sim0; fit a+b·nvs on nearest internal-train label), then augment.
- **S3 weighting** (`weighted_r2`): within-NVS scaffold-grouped CV → per-compound tree-variance std → RF
  `sample_weight = exp(-std/scale)` on NVS rows. (Targets variance, not the systematic bias — expected weakest.)
**Honesty:** `nested_distance` selects the knob on an inner CV and scores on a held-out outer fold; the naive
selection-maximized R² is reported alongside so the optimism gap is visible. All internal folds InChIKey-grouped.
R² = squared Pearson (matches the plot). Mechanics unit-tested on synthetic public SMILES
(`tests/test_nvs_subset_search.py`, 6 checks — no real data).

**mdck pilot RAN and finished (2026-08-28 02:40) → `output/results/20260827_NVS_cellab/`** (levers
`baseline,s1,s5`; taus 0.5/0.6/0.65/0.7/0.768/1.0; `s5_methods shift,affine`; seed 42; 107 internal,
273,236 NVS). Artifacts: `summary.csv` (20 arms), `metrics_dict.pkl`, `preddfs/<arm>.parquet`, `folds.pkl`,
`nvs_fold_distance.parquet`, `train_membership.pkl`, `arms.json`, `README.txt`. Verdict is written up in the
deploy section above: **S1 τ=0.60 wins** (R²det 0.528 vs internal-only 0.474), S5 `shift`/`affine`
bias-correction both HURT. S2/S3/nested were NOT part of this run.
**CAUTION — `summary.csv` bookkeeping on `--resume`:** an arm whose `preddfs/<arm>.parquet` is reused gets an
empty `record_list`, so its `n_train` falls back to 107 and `n_nvs_median` to 0 (see `S1_tau0.700/0.768/1.000`).
The metrics in those rows are still correct (they come from the cached OOF pred_df); only the two count columns
are wrong. Read the true counts from `train_membership.pkl` / `nvs_fold_distance.parquet`.

## Model-family benchmark at NVS scale — LightGBM is the fast option, ElasticNet is not (2026-08-31)

Question: which model family makes a full-NVS subset search affordable (the champion RF needs ~7 min per full
fit, so a tau sweep is expensive)? Two scripts:
- **`python/bench_rf_vs_en.py`** — pure TIMING probe on a **synthetic** sparse-binary matrix of the same shape
  (no chemistry loaded; fit time depends on shape, not values). Times RF / ElasticNet / SGD-elasticnet at
  growing row counts and extrapolates to 273,241 rows.
- **`python/bench_models_transfer.py`** — the REAL measurement, transfer arm: train on the ENTIRE NVS pool,
  predict internal (`--endpoint mdck`). Reports fit seconds + R² (Pearson²) + R²det per family.

**Result (mdck, 273,236 NVS train → 107 internal test; `output/results/bench_models_transfer_mdck.csv`):**
| model | fit_time_s | R²_pears | R²det |
|-------|-----------:|---------:|------:|
| RF champion (200 trees) | 440.5 | 0.407 | −3.569 |
| RF small (50 trees) | 109.9 | 0.412 | −3.719 |
| ElasticNet (scaled) | 1397.1 | 0.283 | −1.456 |
| LightGBM (400×63 leaves) | **33.1** | 0.350 | −3.327 |

Two conclusions:
1. **Pure NVS→internal transfer is bias, not rank** — every family holds R²_pears ≈ 0.28–0.41 while R²det is
   ≈ −1.5 to −3.7. This is the same failure mode the subset search targets, now measured without any internal
   training rows at all. It confirms the 2026-07 source audit (NVS mdck is the wrong assay variant, shifted +1.6–1.9).
2. **For search throughput use LightGBM** — 33 s, ~13× faster than the champion RF, with the same ranking
   quality. **ElasticNet is refuted as the "cheap" option**: on the real dense 273k×4469 matrix it took 1397 s
   (~3× SLOWER than the RF) and ranked worst. The synthetic timing extrapolation in `bench_rf_vs_en.py` does not
   survive contact with the real matrix — trust `bench_models_transfer.py`.
New dependency: `lightgbm==4.7.0` in `requirements.txt` (installed in `ML`; offline, no telemetry).

## Open / next

- **mdck τ=0.60 is not banked yet** — run `python python/run_nvs_cellab.py --levers nested
  --taus 0.5,0.55,0.6,0.65,0.7` for an honest (nested-CV) number + a fold-spread/bootstrap error bar.
  Adopt "internal + nearest-shell NVS (dist<~0.6, ~100 cmpd, RAW)" only if honest R²det ≳ 0.50.
- Run the S2 (scaffold) and S3 (weighting) levers for mdck; then repeat the whole search on the other
  NVS-hurt endpoints (hlm, mlm, caco2).
- Consider swapping the subset-search fitter to **LightGBM** for throughput (13× faster, same ranking).
- **Re-deploy** the 8 RF champions after the InChIKey-grouped folds change — CV metrics and `conf_recal`
  both move; delete the cached metrics pickles first, then restart the webapp so it reads the new bundles.
- Write InChIKey-dedup loader to harmonize core sets to one logS table + report true unique count.
- Convert PharmaBench log10 nM → logS (mol/L) for merge.
- Decide intrinsic-S0 vs apparent-pH handling (pKa conversion vs model intrinsic).
- Manually fetch Wiki-pS0 SI (bRo5 anchor).
