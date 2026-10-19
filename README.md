# ADME_ML

ML models to predict in-house ADME endpoints (solubility, LogD, liver-microsome clearance,
Caco-2 / MDCK permeability with and without a P-gp inhibitor, plasma protein binding) for beyond-Rule-of-5 compounds
(PROTACs / molecular glues), trained on curated public data. See `wiki/wiki.md` for project state.

**Hard rule:** in-house data (SMILES, labels, predictions) never leaves this machine. See `CLAUDE.md`.

---

# ⚑ RETRAIN AFTER NEW DATA — the full checklist

New assay data landed in CDD Vault? Do these four steps in order. Steps 1–3 take a minute. Step 4
starts the two long jobs.

### 1. Bump the run paths

Open `config/config.yaml`. Every path that carries a run date sits in ONE block at the top,
`RUN PATHS`. Change all five to the same stamp:

| key | example for a 2026-09-22 run |
|---|---|
| `ADME_RAW_CSV` | `data/20260922_all_adme.csv` |
| `METRICS_PKL_RF_DIR` | `output/results/20260922_metrics_rf` |
| `METRICS_PKL_CP_DIR` | `output/results/20260922_metrics_cp` |
| `CHEMPROP_OUTPUT_DIR` | `output/predictions_runs_chemprop_transfer_20260922` |
| `CHEMPROP_PRETRAIN_DIR` | `output/chemprop_models/transfer_pretrain_20260922` |

A mixed set silently blends two runs. The worst case: chemprop scores the PREVIOUS run's RF folds.

**CAUTION:** `--overwrite` writes the CDD Vault pull to `ADME_RAW_CSV`. Point that key at a NEW file.
A path that already holds a snapshot is overwritten, and the old internal data is not recoverable.

The last two keys are YAML anchors. `CHEMPROP_TRANSFER.output_dir` and `.pretrain_dir` are aliases of
them, so you edit the value in the `RUN PATHS` block only.

### 2. Nothing to do for the features

`DATA.build_MF_features` caches the PUBLIC block (public SMILES never change) and **always
recomputes the INTERNAL block**. New compounds therefore arrive on their own. Leave
`MF_features_all_path` and its `date` argument alone.

### 3. Decide the chemprop stage-1 pretrain

`assess_predictions_cp` reuses a cached checkpoint per grouping and has no `--force_pretrain` flag.
The new `CHEMPROP_PRETRAIN_DIR` of step 1 forces a fresh stage 1 for every grouping. Keep the old
path only when you know that the new internal compounds add no public twins — leak control drops
public molecules that also exist internally, so a new twin makes the cached checkpoint dirty.

### 4. Run the two jobs — RF first, chemprop second

Chemprop reads RF's folds from the `fold` column of `<METRICS_PKL_RF_DIR>/<endpoint>.pkl`. Start
chemprop only after RF finishes, or it builds its own, incomparable folds.

**CAUTION:** an SSH drop kills an interactive job. Both commands run detached under `screen`.
Watch them with `screen -r rf` / `screen -r cp`, or `tail -f output/logs/20260922_rf.log`.

```bash
# 1. pull the new CDD data + assess RF, all 9 endpoints (detached)
screen -dmS rf bash -c '~/miniconda3/envs/ML/bin/python python/ADME_build_ML.py \
  --overwrite --assess_RF_all_endpoints 2>&1 | tee output/logs/20260922_rf.log'

# 2. only after step 1 finishes — chemprop, 11 models per endpoint (long)
screen -dmS cp bash -c '~/miniconda3/envs/ML/bin/python python/ADME_build_ML.py \
  --assess_CP_all_endpoints 2>&1 | tee output/logs/20260922_cp.log'
```

When both finish, compare the RF and chemprop `r2det` per endpoint, update `webapp.value_model` if a
winner changed, then deploy — see **Useful flags** and **Which model the webapp shows** below.

---

# ⚑ NEW PUBLIC DATASET — features + chemprop pretrain

The checklist above covers new INTERNAL data. A new PUBLIC source needs these steps instead.

### 1. Write the parquet

The live path reads **four columns only** from a public parquet
(`DATA._endpoint_dfs`, `pd.read_parquet(..., columns=[...])`):

| column | content |
|---|---|
| `compound` | a unique id, e.g. `PUB_<endpoint>_<i>`. It must not collide with an internal SRB id. |
| `smiles` | the structure |
| `value` | **already in MODELLING space** — `log10` for solubility/clearance/permeability, `identity` for logd, `logit_pct` for ppb. Match `ADME_ENDPOINTS[<ep>].transform`. |
| `origin` | the provenance label, e.g. `Biogen:hlm_biogen`. `BEST_PUBLIC` selects on this exact string. |

Write it to `autoresearch/predict_adme/public_<name>.parquet` (config `ADME_CACHE`).

You do **not** need a feature block. The older parquets carry ~4,469 feature columns, but that is
legacy from the retired runner. `DATA.build_MF_features` derives every feature from the SMILES.

The existing harmonizer is `autoresearch/predict_adme/build_public_features.py`, which calls
`python/harmonize_adme.py` for the unit conversions and writes the id + the legacy feature block.
**CAUTION:** it reads `ADME_PUBLIC`, `ADME_SCALING` and `BIOGEN_ADME_CSV`, which the lean config no
longer holds. Point it at `config/config_legacy_20260903.yaml`, or convert the units yourself.

### 2. Register the file in the config

```yaml
ENDPOINT_PUBLIC_FILES:
  hlm: [public_hlm.parquet, public_novartis_hlm.parquet, public_<name>.parquet]   # <- add here
BEST_PUBLIC:
  hlm: ['Biogen:hlm_biogen', 'Novartis-NIBR', '<your origin string>']             # <- and here
```

The two keys do different jobs:

| key | gates | effect if you forget it |
|---|---|---|
| `ENDPOINT_PUBLIC_FILES` | **both** RF and chemprop (it builds the wide `df_all`) | the rows never reach any model |
| `BEST_PUBLIC` | **RF only** (`select_best_combo_and_update`) | chemprop uses the new data, RF ignores it |

**CAUTION:** `get_internal_public_sets` drops any origin with **1000 compounds or fewer**
(`--min_n`, default 1000). A small public source therefore never enters an RF arm, whatever
`BEST_PUBLIC` says. Lower `--min_n` for a small source.

### 3. H237 features — usually nothing to do

`DATA.build_MF_features` **tops up the public cache incrementally**. It computes only the compounds
`output/features/<date>_MF_features_H237_public.parquet` does not hold yet, then appends and saves.
A new public file therefore costs one featurization of its own compounds, not a 369k rebuild. The
internal block is always recomputed, and a coverage assertion proves every `df_all` compound has
features.

A **full** recompute is needed only when the H237 DEFINITION changes, not when compounds are added.
Then bump the stamp, which writes a fresh cache under the new name:

```python
data.build_MF_features(params, type='H237', date='20260922')   # full rebuild, ~369k compounds
```

**CAUTION:** `date` is a method argument, **not** a config key. Two places set it — the default in
`python/ADME_build_ML.py:DATA.build_MF_features`, which the CLI uses, and the notebook cell that
calls `data.build_MF_features(params, type='H237', date='20260824')`. Change both, or the CLI and
the notebook read different caches. The descriptastorus block dominates the cost (`n_jobs=32`).

A cached public block whose feature columns no longer match the fresh internal block raises, instead
of concatenating into NaN columns. That assertion names the cache file to rebuild.

### 4. Retrain the chemprop stage-1 pretrain

Stage 1 trains on **public rows only**, per grouping, and never sees an internal label. The
checkpoint is cached at `<CHEMPROP_PRETRAIN_DIR>/<grouping>/` and is reused by every endpoint and
fold of that grouping. New public data does not invalidate it by itself — you must force it.

| path | how to force stage 1 |
|---|---|
| `--assess_CP_all_endpoints` | **no flag exists.** Point `CHEMPROP_PRETRAIN_DIR` (RUN PATHS block) at a new directory. |
| `--deploy_CP_all_endpoints` | `--force_pretrain` |

```bash
ML_PY=~/miniconda3/envs/ML/bin/python

# assess: bump CHEMPROP_PRETRAIN_DIR in config/config.yaml first, then
screen -dmS cp bash -c "$ML_PY python/ADME_build_ML.py --assess_CP_all_endpoints \
  2>&1 | tee output/logs/cp_newpublic.log"

# deploy: the flag does it, no config edit
$ML_PY python/ADME_build_ML.py --deploy_CP_all_endpoints --force_pretrain
```

**CAUTION:** a stale checkpoint ships a model trained on public rows you have since removed, or one
that never saw the rows you just added. Read the log to tell the two apart:

| log line | meaning |
|---|---|
| `> PRETRAIN <n> public rows \| <n> tasks [...] \| <n> ep` | stage 1 ran fresh on that pool |
| `> PRETRAIN reused checkpoint -> <path>` | **the cache was reused** — your new public data did NOT reach stage 1 |
| `> PRETRAIN skipped (df_pretrain is None)` | expected: this is the `internal_cv` arm, which trains from scratch by design |

The deploy path prints the same three lines with a `DEPLOY ` prefix.

### 5. A task with no public labels — `public_proxy`

Stage 1 fits **one target scaler per task**. A task whose public column is entirely empty gets a NaN
scaler, and that head then predicts NaN for ever — the frozen-encoder finetune inherits the scaler and
cannot repair it. `build_ML_data_CP` now asserts this before any training and names the fix.

`CHEMPROP_TRANSFER.public_proxy` fills such a task's PUBLIC rows from a related public column. Internal
rows keep their real labels, so only stage 1 changes, and the stage-2 head re-learns any offset:

```yaml
CHEMPROP_TRANSFER:
  public_proxy:
    mdck_inh: {file: tf_novartis_mdck_perm.parquet, column: LE_MDCKv2_LogPapp}
```

**CAUTION:** a proxy is a modelling assumption, not a measurement. Check that its spread matches the
internal target before you trust it — an offset is harmless (the stage-2 head absorbs it), a different
spread is not. Record every proxy you add in `wiki/wiki.md`.

One pretrain per grouping, so the cost is 7 pretrains, not 9 — `hlm/mlm/rlm` share `clearance`, and
each of `solubility`, `logd`, `caco2`, `mdck`, `mdck_inh`, `ppb` has its own.

---

## Conda environments

Two separate environments — kept apart so the heavy Chemprop/torch stack never disturbs the
core `ML` libraries. Feature caches are written as parquet, which is portable between them.

| env | Python | Used for | Key packages |
|-----|--------|----------|--------------|
| **`ML`** | 3.12+ | Core work, single-task models (XGB/RF), baselines, notebooks | pandas 3.0, scikit-learn 1.8, **xgboost 3.2**, rdkit 2025.9.3 |
| **`chemprop`** | 3.11 | Chemprop multitask D-MPNN, Descriptastorus features, public-data download | **chemprop 2.2**, **descriptastorus 2.8**, **PyTDC 1.1**, torch 2.12, rdkit 2023.9.6 |

```bash
# ML (already present) — core + single-task regressors
conda activate ML

# chemprop (separate; created 2026-07-07) — multitask + Descriptastorus + TDC downloads
conda create -n chemprop python=3.11
conda activate chemprop
pip install chemprop descriptastorus PyTDC
```

Pinned versions in `requirements.txt`. All libraries run fully offline after install;
PyTDC downloads only **public** datasets (Harvard Dataverse), never in-house data.

### Which env runs what (multitask ADME, `autoresearch/predict_adme/`)

- **`chemprop` env** — `build_public_features.py` (needs PyTDC + Descriptastorus) and the
  Chemprop multitask model. Compute Descriptastorus features here.
- **`ML` env** — internal baselines and the single-task **XGBoost** arms (`xgboost` is not in
  the `chemprop` env). RandomForest / ExtraTrees / HistGB arms run in either env.
- **Note:** the two envs have different rdkit versions (2025.9.3 vs 2023.9.6). Keep features
  that feed one model from a single env; MF fingerprint parquets are portable, but rebuild
  rather than mix versions if calibration matters.

## Running the models — `python/ADME_build_ML.py`

One entry point builds and scores everything. Run it in the **`ML`** env; it shells out to the
`chemprop` env for the D-MPNN stages, so you never switch environment yourself
(the binary path is config `CHEMPROP_TRANSFER.chemprop_bin`).

```bash
ML_PY=~/miniconda3/envs/ML/bin/python

# 1) RandomForest — 6 arms per endpoint -> <METRICS_PKL_RF_DIR>/<endpoint>.pkl
$ML_PY python/ADME_build_ML.py --assess_RF_all_endpoints

# 2) Chemprop transfer — 3 arms per endpoint -> <METRICS_PKL_CP_DIR>/<endpoint>.pkl
$ML_PY python/ADME_build_ML.py --assess_CP_all_endpoints
```

The two output directories are config keys `METRICS_PKL_RF_DIR` and `METRICS_PKL_CP_DIR`.

**Run them in that order.** Chemprop reuses RF's folds, which it recovers from the `fold` column of
`<METRICS_PKL_RF_DIR>/<endpoint>.pkl`. If that file does not exist yet, chemprop builds its own
InChIKey-grouped split and prints `NOT FOUND -> INDEPENDENT ... split` — those folds are **not**
comparable to RF fold by fold, and `data.cp_fold_source` records `independent` instead of `rf:<path>`.

Both commands are **compute-or-load**: an endpoint whose pickle already exists is read back, not
retrained. So there are two ways to force a fresh run — delete the pickle, or point the config key at
a new directory. CAUTION: a fresh RF run draws NEW folds, so every chemprop number measured against
the old folds stops being comparable. Re-run both.

### The arms each command produces

| arm | RF (`--assess_RF_all_endpoints`) | chemprop (`--assess_CP_all_endpoints`) |
|-----|----------------------------------|----------------------------------------|
| `internal_cv` | 5-fold CV, internal only | same folds, no pretrain — the whole D-MPNN trains from scratch |
| `augmented_cv` | 5-fold CV, public added to TRAIN only | public pretrain + frozen-encoder finetune per fold |
| `ext_->_internal` | train on public, predict every internal compound | the public-only checkpoint predicts every internal compound |
| `internal_temp`, `augmented_temp`, `ext_->_temp` | 80/20 temporal split | — |

Both models score the **identical folds**: chemprop recovers them from the `fold` column of RF's
pickles (config `METRICS_PKL_RF_DIR`), so the two `r2det` values compare directly. When an RF pickle
is missing, chemprop builds its own InChIKey-grouped split and says so — those folds are **not**
comparable fold by fold.

### Useful flags

```bash
# one endpoint or a few (both assess modes and deploy accept this)
$ML_PY python/ADME_build_ML.py --assess_CP_all_endpoints --endpoints hlm,mlm,rlm

# stricter chemprop leak control: also drop salts / stereoisomers / tautomers of internal molecules
$ML_PY python/ADME_build_ML.py --assess_CP_all_endpoints --leak skeleton

# re-pull the internal experimental data from CDD Vault first
$ML_PY python/ADME_build_ML.py --overwrite --assess_RF_all_endpoints

# fit + register the deployable RF (internal + BEST_PUBLIC) for every endpoint
$ML_PY python/ADME_build_ML.py --deploy_RF_all_endpoints            # add --dry_run to skip MLTrail

# fit + register the deployable chemprop model (public pretrain + frozen finetune) for every endpoint
$ML_PY python/ADME_build_ML.py --deploy_CP_all_endpoints            # add --dry_run to skip MLTrail

# CAUTION: after ANY change to ENDPOINT_PUBLIC_FILES, force stage 1 — the cached pretrain
# checkpoint holds the OLD public pool and would ship a model trained on removed data
$ML_PY python/ADME_build_ML.py --deploy_CP_all_endpoints --force_pretrain
```

| flag | effect |
|------|--------|
| `--config` | YAML to read (default `config/config.yaml`) |
| `--endpoints a,b` | restrict to a subset; default is all 9 |
| `--leak exact\|skeleton\|none` | chemprop only: drop public molecules that also exist internally (default `exact`) |
| `--min_n` | minimum compounds for a public origin to count (default 1000) |
| `--overwrite` | re-pull the CDD Vault export instead of reading the cache |
| `--dry_run` | deploy only: fit and calibrate, do not register to MLTrail |
| `--force_pretrain` | chemprop deploy only: retrain stage 1 instead of reusing the cached grouping checkpoint |

CAUTION: `--assess_CP_all_endpoints` trains **11 chemprop models per endpoint** (1 pretrain,
5 transfer folds, 5 from-scratch folds). Stage 1 is cached per grouping, so `hlm/mlm/rlm` share one
pretrain and `solubility/logd` share another — 7 pretrains across all 9 endpoints. Run it detached:

```bash
screen -S cpall
$ML_PY python/ADME_build_ML.py --assess_CP_all_endpoints 2>&1 | tee output/cp_all.log
```

### Which model the webapp shows

`--deploy_RF_all_endpoints` registers `adme_<ep>_h237` and `--deploy_CP_all_endpoints` registers
`adme_<ep>_cp`. The webapp then reads `webapp.value_model` in the config to decide, per endpoint,
which one supplies the displayed **value**:

| endpoint | value model |
|---|---|
| solubility, logd, hlm, mlm, rlm | `chemprop` (`adme_<ep>_cp`) |
| mdck, caco2, ppb | `rf` (`adme_<ep>_h237`) |

The **confidence** always comes from the RF model, for all 9 endpoints. A chemprop `regression`
head returns no per-row std, so `conf_recal` only exists on the RF side. Deploy RF for every
endpoint even where chemprop supplies the value.

CAUTION: run `--deploy_RF_all_endpoints` before the webapp, whatever `value_model` says. Without
the RF model an endpoint has no confidence and the webapp drops it.

### Deploying ONE endpoint, and refreshing the webapp

Every deployed model is **pinned by id** in config `MLTRAIL_IDS`, one id per endpoint per framework:

```yaml
MLTRAIL_IDS:
  rf:                    # adme_<ep>_h237 — confidence for all 9, value where value_model says rf
    solubility: 20
    ...
  chemprop:              # adme_<ep>_cp — value only, where value_model says chemprop
    mdck_inh:   null     # not deployed yet; the first deploy creates it and prints the id to pin
```

`OUTPUT.vault_id` reads that pin and hands it to `registry.add`, which MLTrail reads as *append a new
version, identity inherited*. `webapp/app.py` binds the same ids, so the deploy and the app always
agree and the vault listing order never matters. A `null` pin means "not deployed yet": the deploy
falls back to an `experiment_name` lookup, creates the entry if there is none, and prints

```
> PIN THIS in config/config.yaml -> MLTRAIL_IDS.chemprop.mdck_inh: 46
```

which you paste back into the config. So an endpoint's id never changes, `--endpoints` is optional,
and re-running a deploy cannot duplicate a vault entry.

Audit the pins against the vault at any time — it loads no data and is instant:

```bash
$ML_PY python/ADME_build_ML.py --check_mltrail_ids
```

It fails if a pinned id is absent from the vault, if it carries the wrong `experiment_name`, or if an
id the webapp needs is still `null`. Exit code 1 means a problem.

CAUTION: an append REFUSES to change an identity field (`experiment_name`, `experiment_measure`, `unit`,
`model_type`, `framework`, `features_type`). Edit `ADME_ENDPOINTS[k].unit` or `DEPLOY.features_type` and
the deploy raises `ValidationError`. That is deliberate — a changed unit means a different model, so
register it under a new name rather than hiding it in a version.

To deploy one endpoint:

```bash
ML_PY=~/miniconda3/envs/ML/bin/python

# fit + calibrate without writing to MLTrail
$ML_PY python/ADME_build_ML.py --deploy_RF_all_endpoints --endpoints mdck_inh --dry_run

# register it
$ML_PY python/ADME_build_ML.py --deploy_RF_all_endpoints --endpoints mdck_inh

# confirm what the vault now holds
$ML_PY -c "from mltrail import Registry; l = Registry.from_default().list(); \
  print(l[l.experiment_name.astype(str).str.startswith('adme_')][['id','experiment_name']].to_string(index=False))"
```

Check the version trail of an endpoint with `Registry.from_default().trail(<id>)`.

History: until 2026-09-24 the RF path passed `model_id=None`, so every deploy minted a duplicate entry.
The 9 duplicates that produced (ids 37-45) were deleted, and the chemprop path moved off
`overwrite=True`, which reset the latest version instead of adding one.

### Running the webapp

```bash
conda run -n ML python webapp/app.py     # -> http://127.0.0.1:8050 (config webapp.host/port)
```

The app binds its models ONCE at startup, by the ids pinned in `MLTRAIL_IDS`, from the vault at
`MLTRAIL_VAULT`. So a redeployed model appears only after a **restart**. Column order comes from
`webapp.endpoint_order`, the triage colour from `webapp.cutoffs`, and which model supplies the value
from `webapp.value_model`.

| config key | effect |
|---|---|
| `MLTRAIL_VAULT` | the vault the deploy writes and the app reads. Its `registry.json` holds absolute paths, so mount it at this same path everywhere |
| `webapp.scripts_dir` | the Scripts clone that holds `ML_Reg` (default `~/Scripts`) |
| `webapp.strict_startup` | `true`: refuse to start when a pinned model does not load, and name it |
| `webapp.run_ttl_min`, `webapp.max_runs` | how long, and how many, scored runs stay available to the SDF export |

Each scored run gets a random `run_id`, and the SDF export serves only the run its id names. So one
user cannot download another user's compounds. The CSV export is built in the browser.
`webapp/requirements.txt` and `webapp/requirements-chemprop.txt` pin both envs. To run the app as a
service on another machine, read `docs/handoff.md`.

CAUTION when testing it from a shell: a multi-line command collapsed onto one line silently chains the
steps and leaves a STALE uvicorn answering. Put webapp test sequences in a script file and run
`bash script.sh`.

### The same steps from the notebook

`vignettes/Multitask_adme_preds.ipynb` calls the same class methods, one endpoint at a time, so you
can inspect each frame:

```python
params = PARAMS('config/config.yaml').load_params()
data   = DATA(); data.load_df_internal_exp_all(params)
data.load_combine_dfs(params)          #_> data.df_all (WIDE: 1 row/compound x 8 endpoint columns)
data.build_MF_features(params)         #_> data.MF_features['all'] (public cached + internal recomputed)
output = OUTPUT(params)

data.build_ML_data_RF(params, k=k)     #_> data.ML_data[k]           (RF frame)
data.build_ML_data_CP(params, k=k)     #_> data.cp_pub / cp_int / cp_folds  (chemprop pools)
output.assess_predictions(data, METRICS_PKL)                        #_> output.metrics_results[k]
output.assess_predictions(data, METRICS_PKL_CP, ml_model='chemprop') #_> output.metrics_results_cp[k]
```

### Config

`config/config.yaml` serves exactly three consumers — `python/ADME_build_ML.py`, the vignette
notebook, and `webapp/app.py`. Its header lists which keys each one reads. The measured scores live
in `wiki/wiki.md`, never in the config, so they cannot go stale in silence.

### Running the vignette notebook (`vignettes/Multitask_adme_preds.ipynb`)

NOTE (2026-09-08): the notebook now runs in the **`ML`** env and shells out to the `chemprop`
binary, exactly like the CLI above — see the section above. The `chemprop`-env instructions below
apply only to the older in-process Chemprop cells.

The notebook used to be the single place both RF (sklearn) and Chemprop models are trained, run from
the **`chemprop`** env. A fresh `chemprop` env is missing four libraries the top
import cell pulls in — install them once (versions pinned in `requirements.txt`, taken from the
working `ML` env; no rdkit/torch changes):

```bash
conda activate chemprop
pip install xgboost==3.2.0 openTSNE==1.0.4 py3Dmol==2.5.3 meeko==0.7.1
```

- **xgboost, openTSNE** — ML (single-task XGB arms, chemical-space embedding plots).
- **py3Dmol, meeko** — 3D viewer / docking prep; imported at the top but unused in the ADME
  modelling sections. If `pip` tries to touch rdkit/torch, add `--no-deps` for `meeko` (or drop
  those two and comment their imports — the modelling cells don't need them).

### RAPIDS (GPU cuRF) env — optional, for large-row RF fits

The single-task RandomForest fits can run on GPU via RAPIDS **cuML** (`cuRF`). It only pays off
for **large-row** arms — e.g. the MDCK/Novartis augmented set (~336k rows). On small-row,
high-feature arms (e.g. LogD, ~4k rows × 4,469 features) 32-core sklearn is faster. Enable it
**per run** with `use_cuml=True` on `transfer()` / `predict_and_record()` in the notebook;
sklearn stays the default (`make_model()` builds either from the same `champion` params).

Caveats when `use_cuml=True`: no `uq_std` / `confidence` (cuML has no per-tree API, so those come
back NaN), and results are not bit-identical to sklearn (float32 + `n_streams`). Keep sklearn as
the reference champion.

Create the env (RAPIDS core is conda-only; tested on CUDA 12.9 / RTX A6000):

```bash
# 1) RAPIDS core (resolves the newest cuML compatible with the CUDA 12.9 driver)
conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
    cuml=26.08 python=3.12 'cuda-version>=12.0,<=12.9'

# 2) extras the notebook's top import cell needs (absent from a fresh rapids env)
conda install -n rapids -c conda-forge networkx requests adjusttext xgboost \
    openTSNE pyyaml dill seaborn pyarrow ipykernel
conda run -n rapids pip install nonconformist          # pip-only

# 3) register the Jupyter kernel FROM the env so activation runs (cupy needs CUDA on the path)
conda run -n rapids python -m ipykernel install --user --name rapids --display-name "Python (rapids)"
```

Pinned versions in `requirements_rapids.txt`. **Run the notebook on the "Python (rapids)"
kernel** — pointing Jupyter directly at the env's bare `python` skips conda activation and cupy
then fails with "Failed to find CUDA headers". (`pip` alternative for the core:
`pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12`.)
