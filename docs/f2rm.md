# f2rm — files to keep, files to remove

Checked on 2026-10-19. The check used file names, sizes, code and config only. It did not open a data file.

**STATUS: done on 2026-10-19.** The user ran `delete_old_files.sh` with 40 `YOUR CALL` lines turned on.
Every `safe to delete` path and those 40 paths are gone. The 17 kept `YOUR CALL` paths stay as commented
lines in the script. `wiki/wiki.md` ("Repo cleanup done") lists what went and what can no longer run.
The KEEP sections below still say why each kept file matters.

## What "needed" means

A file is needed when one of these three targets reads it:

1. **webapp** — `webapp/app.py`, the drag-and-drop scorer.
2. **build** — `python/ADME_build_ML.py` in all its CLI modes: `--overwrite`, `--assess_RF_all_endpoints`,
   `--assess_CP_all_endpoints`, `--deploy_RF_all_endpoints` and `--deploy_CP_all_endpoints`.
3. **nb** — every code cell of the 5 notebooks in `vignettes/`. `nb:<name>` names the notebook.

## Tags

| tag | meaning |
|---|---|
| `KEEP` | a target reads it |
| `KEEP (cache)` | a target reads it to save time; the code can make it again, but slowly |
| `IN USE` | the RF run of 2026-10-19 (screen `gene_SAR`) writes it; do not touch it until the run ends |
| `YOUR CALL` | no target reads it, but it is the test suite, a rebuild tool, or the only copy of something |
| `safe to delete` | no target reads it: an old output, a backup, or a retired experiment |

CAUTION: `data/`, `output/`, `autoresearch/` and `tmp/` are not in git. A file that you delete there is gone.
`git checkout` can restore a deleted file from the code folders.

**`safe to delete` frees about 44 GB** (as `du -h` counts): `autoresearch/` 39.3 GB, `output/` 4.5 GB,
`data/` 0.2 GB. Two folders hold 36 GB of it: `autoresearch/predict_adme/chemprop_run/` and `chemprop_transfer_run/`.

**To delete:** edit and run `bash delete_old_files.sh` (repo root). It has one `rm -rf` line per path:
117 active lines (`safe to delete`) and 57 lines that start with `#` (`YOUR CALL`). Remove a `#` to delete
that path too. The script works in its own folder and stops when that folder is not the ADME_ML root.

## Two notebook cells that cannot run now

- `nb:Multitask_adme_preds`, cell `01a26a07` (round trip), reads `tmp/20260909_pred_internal.csv`. That file
  is gone. Cell `eb77e860` uses the output of `01a26a07`, so it fails too. To repair them, export the internal
  set from the webapp again and point the cell at the new file.
- `nb:Solubility_sticky_cmps`, cell `7b74aef2`, used `CHEMLIB_PATH` and `CHEMLIB_OVERWRITE`, which no cell
  defined. **FIXED 2026-10-19:** both are cell constants now, and `CHEMLIB_PATH` reads
  `data/chemical_libs/SERAC_lib.csv`, so that file is KEEP.

## Repo root (in git)

```text
CLAUDE.md                   # KEEP — instructions for Claude
README.md                   # KEEP — documentation
requirements.txt            # KEEP — packages of env ML
requirements_rapids.txt     # YOUR CALL — only the GPU option make_model(use_cuml=True)
remove_adme_ml.sh           # safe to delete — one-off vault cleanup; the vault holds no superseded adme model now (18 of 18 current)
.gitignore                  # KEEP
.claude/                    # KEEP — settings, incl. the hook that blocks Read on a notebook
```

## archive/ (in git)

```text
archive/                    # safe to delete — the superseded chemprop runner; nothing imports it
    run_chemprop_transfer.py
    run_cp_transfer.sh
```

## autoresearch/ (41 GB, not in git)

```text
autoresearch/
    predict_adme/                                   # build reads the public files here (config ADME_CACHE)
        public_solubility.parquet                   # KEEP — build (config ENDPOINT_PUBLIC_FILES)
        public_logd.parquet                         # KEEP — build; nb:Visualize_fingerprints
        public_hlm.parquet                          # KEEP — build
        public_rlm.parquet                          # KEEP — build
        public_caco2.parquet                        # KEEP — build
        public_ppb.parquet                          # KEEP — build
        public_chembl_mdck_inh.parquet              # KEEP — build
        public_admetlab_caco2.parquet               # KEEP — build
        public_novartis_hlm.parquet                 # KEEP — build (185 MB)
        public_novartis_mlm.parquet                 # KEEP — build (185 MB)
        public_novartis_rlm.parquet                 # KEEP — build (185 MB)
        public_novartis_caco2.parquet               # KEEP — build (185 MB)
        public_novartis_mdck.parquet                # KEEP — build (185 MB)
        public_novartis_ppb.parquet                 # KEEP — build (185 MB)
        tf_novartis_mdck_perm.parquet               # KEEP — build, chemprop: mdck_inh proxy + mdck_perm grouping
        tf_novartis_ppb_fu.parquet                  # KEEP — build, chemprop: ppb_fu grouping
        _ikcache/                                   # KEEP (cache) — InChIKeys for leak control (fn.inchikeys_for)
        internal_targets.parquet                    # YOUR CALL — only the test fixture (tests/_adme_fixture.py)
        internal_MF.parquet                         # YOUR CALL — only the test fixture
        *.py, *.sh (26 scripts)                     # YOUR CALL — retired autoresearch code, NOT in git (only copy)
        chemprop_run/                               # safe to delete — 28 GB, 2026-07 chemprop runs
        chemprop_transfer_run/                      # safe to delete — 8.4 GB, 2026-07 chemprop transfer runs
        logs/                                       # safe to delete — 112 MB, autoresearch logs
        public_admetlab_logd.parquet                # safe to delete — not in ENDPOINT_PUBLIC_FILES
        public_admetlab_mdck.parquet                # safe to delete — not in ENDPOINT_PUBLIC_FILES
        public_admetlab_ppb.parquet                 # safe to delete — not in ENDPOINT_PUBLIC_FILES
        public_admetlab_solubility.parquet          # safe to delete — not in ENDPOINT_PUBLIC_FILES
        public_novartis_logd.parquet                # safe to delete — not in ENDPOINT_PUBLIC_FILES (185 MB)
        public_solubility.contaminated_bak.parquet  # safe to delete — backup from before the 2026-07 cleaning
        public_solubility_full.parquet              # safe to delete — copy from before the 2026-07 cleaning
        internal_DS.parquet                         # safe to delete — only the retired systematic runner reads it
        novartis_ikey.parquet                       # safe to delete — no reader
        tf_ds_cache.parquet                         # safe to delete — only two retired scripts read it (163 MB)
        __pycache__/                                # safe to delete
    predict_adme_raw/                               # safe to delete — 1.6 GB, raw-units experiment
    predict_adme_solclean/                          # safe to delete — 71 MB, solubility cleaning sweep
    predict_solubility/
        training_MF_features.parquet                # KEEP — nb:Solubility_sticky_cmps reads it when SOL_OVERWRITE = False
        test_MF_features.parquet                    # KEEP — nb:Solubility_sticky_cmps, same
        training.parquet                            # safe to delete — written when SOL_OVERWRITE = True, never read
        test.parquet                                # safe to delete — same
        *.py (4 scripts), config.yaml               # YOUR CALL — retired classifier code, NOT in git (only copy)
        logs/                                       # safe to delete — 989 MB
        __pycache__/                                # safe to delete
```

## config/ (in git)

```text
config/
    config.yaml                     # KEEP — all three targets
    chembl_harvest.yaml             # YOUR CALL — only harvest_chembl_mdck_inh.py (rebuilds public_chembl_mdck_inh.parquet)
    config_legacy_20260903.yaml     # YOUR CALL — only harmonize_adme.py / build_public_features.py (rebuild the public parquets)
```

## data/ (292 MB, not in git)

```text
data/
    20261019_all_adme.csv                   # KEEP — build + nb (config ADME_RAW_CSV), the 2026-10-19 CDD pull
    20260923_all_adme.csv                   # YOUR CALL — the CDD pull behind the deployed models; only copy
    20260707_all_adme.csv                   # YOUR CALL — the 2026-07-07 CDD pull; only copy
    20260625_thermoSol.csv                  # KEEP — nb:Solubility_sticky_cmps
    FP/                                     # KEEP — nb:FP_preds (20260914_U1.csv, 20260914_U2.csv)
    sticky_compounds/                       # KEEP — nb:Solubility_sticky_cmps (sticky_compounds.xlsx)
    public_solubility/                      # nb:Solubility_sticky_cmps runs harmonize_solubility() on these
        chembl_api/                         # KEEP
        aqsoldbc_solcuration/               # KEEP
        pharmabench/                        # KEEP
        biogen_fang_adme/                   # KEEP
        wiki_ps0_bigmol/                    # KEEP
        solubility_challenge/               # KEEP
        protac_patent/                      # KEEP (86 MB)
        aqsoldb/                            # safe to delete — harmonize_sol.py does not read it
        esol_delaney/                       # safe to delete — harmonize_sol.py does not read it
        llompart_curated/                   # safe to delete — harmonize_sol.py does not read it
        harmonized_sol.csv                  # safe to delete — the notebook writes it again on each run (26 MB)
    public_adme/                            # YOUR CALL — raw TDC + ChEMBL downloads; only the rebuild scripts read them
    chemical_libs/                          # KEEP — nb:Solubility_sticky_cmps reads SERAC_lib.csv (CHEMLIB_PATH)
    protacdb2.0_zinc_chembl_dataset.csv     # safe to delete — 131 MB; only a commented-out line names it
```

## docs/ (in git)

```text
docs/
    handoff.md      # KEEP — webapp server handoff
    f2rm.md         # this file
```

## output/ (11 GB, not in git)

```text
output/
    features/
        20260824_MF_features_H237_public.parquet    # KEEP (cache) — build + nb, public H237 block (236 MB); a rebuild takes long
        20260824_MF_features.parquet                # safe to delete — legacy matrix (244 MB), read only when the public cache is missing
        internal_H236.parquet                       # safe to delete — only the retired systematic runner reads it
    results/
        20261019_metrics_rf/                        # IN USE — the running RF job (config METRICS_PKL_RF_DIR)
        20260923_metrics_rf/                        # KEEP — nb:ADME_ML_results (cell b241853a, D = '20260923')
        20260923_metrics_cp/                        # KEEP — nb:ADME_ML_results, same
        20260909_metrics_rf/                        # KEEP — nb:ADME_ML_results (cell d8e042dc, D = '20260909')
        20260909_metrics_cp/                        # KEEP — nb:ADME_ML_results, same
        20260831_ADME_nearshell/                    # KEEP — default nearshell_dir of fn.plot_bar_summary_RF_CP_all_endpoints
        20260825_metrics/                           # safe to delete — older RF run
        20260907_metrics_cp/                        # safe to delete — older chemprop run
        20260827_NVS_campaign/                      # safe to delete — NVS experiment
        20260827_NVS_cellab/                        # safe to delete — NVS experiment
        bench_models_transfer_mdck.csv              # safe to delete
        tmp/                                        # safe to delete — only commented-out notebook lines name it
    chemprop_models/
        transfer_pretrain/                          # stage-1 checkpoints (config CHEMPROP_PRETRAIN_DIR)
            all8/ clearance/ mdck_perm/ permeability/ permeability_inh/ ppb_fu/ sol_lipo/
                                                    # KEEP (cache) — build reuses them; without them each grouping trains stage 1 again
            clearance_smoke/ mdck_perm_smoke/ ppb_fu_smoke/
                                                    # safe to delete — smoke-test leftovers
        all8/ clearance/ sol_logd/                  # safe to delete — 2026-07 multitask models (0.8 GB)
    predictions_runs_chemprop_transfer/
        summary_transfer.csv                        # KEEP — nb:Multitask_adme_preds (cell 9fb460e4)
        (all the rest)                              # safe to delete — 2.3 GB of old workdirs, deploy_* and logs
    predictions_runs_chemprop_transfer_20260923/    # safe to delete — 1.2 GB, workdirs of the 2026-09-23 chemprop run
    figures/                                        # YOUR CALL — your saved figures; no code reads them
    ppt/                                            # YOUR CALL — your slides
    logs/                                           # safe to delete — logs of the 2026-09 deploy and check runs
    predictions_runs/ predictions_runs_admesweep/ predictions_runs_chemprop/ predictions_runs_raw/
    predictions_runs_solclean/ predictions_runs_solsweep/ predictions_runs_temporal/
    predictions_runs_temporal_fractions/            # safe to delete — 2026-07 outputs of retired runners; the wiki keeps their numbers
    _cp_firstpass_backup/                           # safe to delete — backup
    chemprop_hpo_best.json                          # safe to delete — no reader
    train_counts.csv train_counts_fractions.csv     # safe to delete — their only reader was a notebook that is gone
    *.log (13 files at this level)                  # safe to delete — 2026-07 run logs
```

The webapp needs nothing in `output/`. All 18 pinned models (RF and chemprop) live inside the vault.
The next chemprop run creates `results/20261019_metrics_cp/` and `predictions_runs_chemprop_transfer_20261019/`.

## python/ (in git)

```text
python/
    ADME_build_ML.py                    # KEEP — build; the notebooks import PARAMS / DATA / OUTPUT
    functions.py                        # KEEP — build + nb
    harmonize_sol.py                    # KEEP — the nb import cells; nb:Solubility_sticky_cmps calls it
    nb_edit.py                          # KEEP — the notebook-edit tool that CLAUDE.md requires
    run_RF_SingleTask_systematic.py     # YOUR CALL — only the tests import it
    run_Chemprop_SystematicGroups.py    # YOUR CALL — only the tests import it
    nvs_subset_search.py                # YOUR CALL — the tests + run_adme_nearshell.py import it
    build_novartis_clusters.py          # YOUR CALL — rebuilds tf_novartis_*.parquet
    harmonize_adme.py, extract_adme.py  # YOUR CALL — rebuild the public parquets
    harvest_chembl_mdck_inh.py          # YOUR CALL — rebuilds public_chembl_mdck_inh.parquet
    run_adme_nearshell.py, nvs_campaign.py
                                        # YOUR CALL — made results/20260831_ADME_nearshell
    make_datasets_slide.py, make_potency_table_slide.py, ppt_style.py
                                        # YOUR CALL — your slide builders (output/ppt)
    adme_cleaning_sweep.py  adme_train_counts.py  bench_models_transfer.py  bench_rf_vs_en.py
    build_fraction_counts.py  combine_public_to_internal.py  compute_metrics_sweep.py
    raw_units_experiment.py  run_nvs_cellab.py  sanitize_solubility.py  solubility_cleaning_sweep.py
    temporal_eval.py  temporal_fractions_chemprop.py  temporal_fractions_rf.py
                                        # safe to delete — retired experiments; no target and no test imports them
    __pycache__/                        # safe to delete
```

## tests/ (in git)

```text
tests/              # YOUR CALL — the unit-test suite; no target needs it. It needs the 3 test modules
                    # marked above and autoresearch/predict_adme/internal_targets.parquet + internal_MF.parquet
    __pycache__/    # safe to delete
```

## tmp/ (not in git)

```text
tmp/
    20260910_intern_preds.csv                       # KEEP — nb:Multitask_adme_preds (cell 3c91817f)
    20260923_intern_preds.csv                       # KEEP — nb:Multitask_adme_preds (cell f7e1632f)
    test_cmpds.sdf, test_cmps.csv                   # YOUR CALL — no code reads them; test uploads for the webapp?
    20260910_intern_input.csv                       # safe to delete — no reader
    20260916_intern_preds.csv                       # safe to delete — no reader
    Multitask_adme_preds.ipynb.bak_20260904_083143  # safe to delete — old notebook backup
    wiki.md.bak_20260902_101612                     # safe to delete — old wiki backup
    to_delete/                                      # safe to delete — empty
```

## vignettes/, webapp/, wiki/ (in git)

```text
vignettes/                          # KEEP — the 5 notebooks
webapp/
    app.py                          # KEEP — webapp
    static/                         # KEEP — webapp (index.html, app.js)
    requirements.txt                # KEEP — server env (see docs/handoff.md)
    requirements-chemprop.txt       # KEEP — server env for the chemprop values
    __pycache__/                    # safe to delete
wiki/                               # KEEP — the project memory
```

## Outside the repo — do not delete

These are not repo files, but the targets fail without them:

```text
~/Scripts/
    ML_Reg.py, Statistics_tools.py  # webapp + build + nb
    Rdkit_tools.py, ML_Class.py     # build + nb
    Molecule.py                     # nb
    MolViz3D.py                     # nb:Solubility_sticky_cmps
~/CDD_Vault_API/                    # build --overwrite + nb (get_protocol_data, get_library, config, token)
/data2/MLTrail_vault/               # build deploys here; the webapp and the confidence cell read it
~/miniconda3/envs/ML, envs/chemprop # the two Python envs
```
