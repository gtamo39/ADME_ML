#!/usr/bin/env bash
# Chemprop transfer arm — specialist pretrain groupings, run one after the other.
#
# Tests the "task relatedness decides" hypothesis: `clearance` (one assay, 3 species) beat RF on 3/3
# endpoints, while `all8` (8 heterogeneous properties) lost 5/5. Each step below pretrains a COHERENT
# block on public data, then frozen-encoder finetunes on RF's exact folds.
#
# Only the `transfer` arm runs — the `scratch` controls are already in summary_transfer.csv and do not
# depend on the grouping. Results accumulate keyed by (endpoint, arm, grouping), so an endpoint ends up
# with several groupings side by side. Each step is --resume, so a rerun skips finished work.
#
#   bash run_cp_transfer.sh                        # foreground, in a screen
#   setsid nohup bash run_cp_transfer.sh >/dev/null 2>&1 &    # fully detached
set -u
ROOT=/home/gtamo/ADME_ML; cd "$ROOT"
ML_PY=$HOME/miniconda3/envs/ML/bin/python
LOGDIR=output/predictions_runs_chemprop_transfer; mkdir -p "$LOGDIR"
PROG="$LOGDIR/progress_transfer.log"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
step(){ echo "[$(ts)] $*" | tee -a "$PROG"; }

# one step = one (endpoints, grouping) pair; a failure is logged and the next step still runs
run_step(){
  local name=$1 eps=$2 grp=$3
  step "--- START $name : endpoints=$eps grouping=$grp"
  local t0=$SECONDS
  $ML_PY python/run_chemprop_transfer.py --config config/config.yaml \
      --endpoints "$eps" --grouping "$grp" --arms transfer --resume \
      >"$LOGDIR/run_${name}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    step "--- DONE  $name ($((SECONDS - t0))s)"
    grep -E '^> [a-z]+ +\[transfer' "$LOGDIR/run_${name}.log" | tee -a "$PROG"
  else
    step "--- FAILED $name (exit $rc) — see $LOGDIR/run_${name}.log"
    tail -5 "$LOGDIR/run_${name}.log" | tee -a "$PROG"
  fi
}

step "===== CHEMPROP TRANSFER — specialist groupings START (pid $$) ====="
$ML_PY -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | tee -a "$PROG"

# permeability block (caco2 + mdck): the direct clearance analogue, same Papp A->B assay family
run_step permeability   caco2,mdck        permeability
# mdck + its related raw Novartis permeability columns as auxiliary tasks (6 tasks)
run_step mdck_perm      mdck              mdck_perm
# ppb + its free-fraction family (11 tasks): the biggest all8 failure (-0.660) and the most headroom
run_step ppb_fu         ppb               ppb_fu
# solubility + logd as a PRETRAIN block (the 2026-07 "don't ship" verdict was for multitask prediction)
run_step sol_lipo       solubility,logd   sol_lipo

step "===== CHEMPROP TRANSFER — specialist groupings DONE ====="
step "summary -> $LOGDIR/summary_transfer.csv"
# final table: every (endpoint, arm, grouping) scored so far
$ML_PY - <<'EOF' 2>&1 | tee -a "$PROG"
import pandas as pd
t = pd.read_csv('output/predictions_runs_chemprop_transfer/summary_transfer.csv')
print("\n===== ALL ARMS SO FAR (R2det, RF's exact folds) =====")
print(t.sort_values(['endpoint', 'r2det'], ascending=[True, False]).to_string(index=False))
w = t[t.arm == 'transfer'].pivot_table(index='endpoint', columns='grouping', values='r2det')
print("\nR2det by pretrain grouping (transfer arm):")
print(w.to_string())
EOF
