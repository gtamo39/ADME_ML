# ADME_ML — Project Wiki

Durable aggregate memory for this repo. Survives context compaction. Aggregate-only (no SMILES / compound IDs / per-compound values).

---

## Current objective

Train a public-data ML model to predict **thermodynamic solubility** of in-house compounds (PROTACs / molecular glues, bRo5).
- **In-house target:** `data/20260625_thermoSol.csv` — thermodynamic solubility in μM, range ~1.5 → 21,000 μM (4+ orders of magnitude → model in log space). Censored `< X` values parsed to float in `vignettes/Main.ipynb`.
- **Feature/compound reference:** `data/protacdb2.0_zinc_chembl_dataset.csv` (ProtacDB 2.0 + ZINC + ChEMBL, with predicted ADME columns — clearance, LogFu, LogP/D, permeability, CYP; **no solubility column**).
- **Strategy (2026-06-25):** pretrain on curated thermodynamic small-molecule sets → fine-tune on in-house μM data. Multi-task option using Biogen-Fang ADME endpoints.

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

## Open / next

- Write InChIKey-dedup loader to harmonize core sets to one logS table + report true unique count.
- Convert PharmaBench log10 nM → logS (mol/L) for merge.
- Decide intrinsic-S0 vs apparent-pH handling (pKa conversion vs model intrinsic).
- Manually fetch Wiki-pS0 SI (bRo5 anchor).
