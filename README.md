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

### Running the vignette notebook (`vignettes/Multitask_adme_preds.ipynb`) in `chemprop`

The notebook is the single place both RF (sklearn) and Chemprop models are trained/evaluated, so
run it from the **`chemprop`** env. A fresh `chemprop` env is missing four libraries the top
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
