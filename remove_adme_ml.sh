#!/usr/bin/env bash
# Delete the SUPERSEDED ADME models from the MLTrail vault:
#   * the H236 single-task RF models (adme_<ep>), replaced by adme_<ep>_h237
#   * the previous multitask chemprop models (adme_mt_clearance, adme_mt_all8)
# The H237 RF champions and any new adme_<ep>_cp chemprop model STAY.
# CAUTION: --apply deletes the model, its whole version trail, its model artifacts and its archived
# training set. There is NO backup. This cannot be undone.
# Usage:
#   ./remove_adme_ml.sh            # DRY RUN: show what it would delete, change nothing
#   ./remove_adme_ml.sh --apply    # really delete
set -euo pipefail

ML_PY=~/miniconda3/envs/ML/bin/python
MLTRAIL=~/miniconda3/envs/ML/bin/mltrail
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
TARGETS=$(mktemp)
trap 'rm -f "$TARGETS"' EXIT

# select the targets and refuse to run if an endpoint would lose its last model
"$ML_PY" - "$TARGETS" <<'PY'
import sys, yaml, pandas as pd, mltrail

reg = mltrail.Registry.from_default()
rows = []
for i in reg.list().id:
    d = reg.details(int(i))
    rows.append({'id': int(i), 'name': d.get('experiment_name'), 'framework': d.get('framework'),
                 'features_type': d.get('features_type')})
df = pd.DataFrame(rows)
adme = df[df.name.str.startswith('adme_', na=False)]

# rule 1: the old single-task RF models, built on the H236 fingerprint block only
h236 = adme[(adme.framework == 'sklearn') & (adme.features_type == 'H236')]
# rule 2: the previous multitask chemprop models; a new adme_<ep>_cp model is NOT a target
cp = adme[(adme.framework == 'chemprop') & ~adme.name.str.endswith('_cp')]
tgt = pd.concat([h236, cp]).sort_values('id')

# safety: every configured endpoint must keep an H237 sklearn model after the delete
keep = adme[~adme.id.isin(set(tgt.id))]
eps = list(yaml.safe_load(open('config/config.yaml'))['ADME_ENDPOINTS'])
orphan = [e for e in eps
          if keep[(keep.name == f'adme_{e}_h237') & (keep.features_type == 'H237')].empty]
if orphan:
    sys.exit(f'ABORT: these endpoints would keep no H237 model: {orphan}')

print(f'\n> KEEP ({len(keep)} adme model(s)):')
print(keep.to_string(index=False) if len(keep) else '  (none)')
print(f'\n> DELETE ({len(tgt)} adme model(s)):')
print(tgt.to_string(index=False) if len(tgt) else '  (none)')
cfg = mltrail.default_config()
print(f"\n> vault: {cfg['registry_path']}")
# list the artifacts + training-set folder each target owns, so the dry run shows every file that goes
import glob, os, subprocess
paths = [p for i in tgt.id
         for p in glob.glob(os.path.join(cfg['trained_models_dir'], f'{i}_v*'))
         + [os.path.join(cfg['training_sets_dir'], str(i))] if os.path.exists(p)]
if paths:
    du = subprocess.run(['du', '-shc', *paths], capture_output=True, text=True).stdout
    print(f'> vault files to remove ({len(paths)}):')
    print(''.join(f'  {l}\n' for l in du.strip().splitlines()))
open(sys.argv[1], 'w').write('\n'.join(str(i) for i in tgt.id))
PY

IDS=$(tr '\n' ' ' < "$TARGETS")
[[ -z "${IDS// }" ]] && { echo -e "\n> nothing to delete — the vault is already clean."; exit 0; }

if [[ $APPLY -eq 0 ]]; then
  echo -e "\n> DRY RUN — nothing changed. Re-run with --apply to delete ids:$IDS"
  exit 0
fi

# delete one model at a time so a failure stops the loop and leaves the rest intact
for i in $IDS; do
  echo "> deleting model id $i"
  "$MLTRAIL" --delete --id "$i"
done

echo -e "\n> remaining adme models:"
"$ML_PY" -c "
import mltrail
reg = mltrail.Registry.from_default()
for i in reg.list().id:
    d = reg.details(int(i))
    if str(d.get('experiment_name')).startswith('adme_'):
        print(f\"  {i:>3}  {d.get('experiment_name'):22s} {d.get('framework'):9s} {d.get('features_type')}\")
"
