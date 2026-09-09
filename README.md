# ADME_ML

ML models to predict in-house ADME endpoints (solubility, LogD, liver-microsome clearance,
Caco-2 / MDCK permeability, plasma protein binding) for beyond-Rule-of-5 compounds
(PROTACs / molecular glues), trained on curated public data. See `wiki/wiki.md` for project state.

**Hard rule:** in-house data (SMILES, labels, predictions) never leaves this machine. See `CLAUDE.md`.

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
| `--endpoints a,b` | restrict to a subset; default is all 8 |
| `--leak exact\|skeleton\|none` | chemprop only: drop public molecules that also exist internally (default `exact`) |
| `--min_n` | minimum compounds for a public origin to count (default 1000) |
| `--overwrite` | re-pull the CDD Vault export instead of reading the cache |
| `--dry_run` | deploy only: fit and calibrate, do not register to MLTrail |
| `--force_pretrain` | chemprop deploy only: retrain stage 1 instead of reusing the cached grouping checkpoint |

CAUTION: `--assess_CP_all_endpoints` trains **11 chemprop models per endpoint** (1 pretrain,
5 transfer folds, 5 from-scratch folds). Stage 1 is cached per grouping, so `hlm/mlm/rlm` share one
pretrain and `solubility/logd` share another — 6 pretrains across all 8 endpoints. Run it detached:

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

The **confidence** always comes from the RF model, for all 8 endpoints. A chemprop `regression`
head returns no per-row std, so `conf_recal` only exists on the RF side. Deploy RF for every
endpoint even where chemprop supplies the value.

CAUTION: run `--deploy_RF_all_endpoints` before the webapp, whatever `value_model` says. Without
the RF model an endpoint has no confidence and the webapp drops it.

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
