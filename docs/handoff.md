# ADME prediction webapp — deployment handoff

**Audience:** the engineer who will run this app as a service on a company web server, and their
AI assistant. You do not need to know the modelling work. You need to know what the process reads,
what it writes, and how it behaves with more than one user.

**Source of truth for the science:** `wiki/wiki.md` and `README.md` in this repo. This file covers
*operations only*.

**Status (2026-10-19):** the four code changes that a shared server needs are done and verified —
per-run download ids, strict startup, configurable paths, and no temp-folder leak. The work that
remains is on the server side (section 12).

---

## 1. What the app is

`webapp/app.py` is a FastAPI service with a static JavaScript front end. A chemist drags an `.sdf`
or `.csv` file onto the page. The app predicts 9 ADME properties per compound, renders a grid, and
exports the selection as CSV or SDF.

```
browser  --multipart upload-->  FastAPI  --read-->  /data2/MLTrail_vault (models)
         <--NDJSON stream----            --exec-->  chemprop CLI (separate conda env)
```

There is **no database and nothing is written to disk**. Each scored run stays in memory for at
most 60 minutes so that its SDF export can find the original records (section 6.1).

### Data classification — read this before choosing where it runs

The app receives, renders and returns **chemical structures and predicted property values**. In this
project that is confidential data. The app binds `127.0.0.1`. On a server, reach it only through an
authenticating reverse proxy on the same host (sections 6.3 and 8).

### If you are an AI assistant working on this server

`CLAUDE.md` at the repo root holds this project's data rules, and they apply on the server too. Your
tool output crosses to your model provider, so:

- Do not read uploaded files, the upload staging folder, or downloaded exports.
- Do not read `/data2/MLTrail_vault/training_sets/`. It archives the training structures and labels.
- Read schemas, not rows. Column names and shapes are fine; values are not.
- Smoke-test with public reference structures only: ethanol `CCO`, benzene `c1ccccc1`, caffeine.
- `registry.json` and each model's `deploy_meta.json` hold metadata only, and are fine to inspect.
- Do not open `*.ipynb` files with a file-read tool. `.claude/settings.json` blocks that; extract
  cell sources with a script instead.

---

## 2. Hard requirements

| # | What | Source | Config key |
|---|---|---|---|
| 1 | `ML` conda env | `webapp/requirements.txt` | — |
| 2 | `chemprop` conda env | `webapp/requirements-chemprop.txt` | `CHEMPROP_TRANSFER.chemprop_bin` |
| 3 | Shared helpers | `git clone git@github.com:gtamo39/Scripts.git` | `webapp.scripts_dir` |
| 4 | MLTrail package | `git clone git@github.com:gtamo39/MLTrail.git`, then `pip install -e` | — |
| 5 | The model vault | read-only mount of `/data2/MLTrail_vault` | `MLTRAIL_VAULT` |

All config keys live in this repo's `config/config.yaml`.

### 2.1 The `ML` environment

```bash
conda create -n ML python=3.12.13
conda run -n ML pip install -r webapp/requirements.txt
conda run -n ML pip install -e <clone of MLTrail>
```

`webapp/requirements.txt` pins every package to the version tested on 2026-10-19. Three pins are not
negotiable:

| package | version | why exact |
|---|---|---|
| scikit-learn | 1.8.0 | the 9 RF models in the vault are joblib pickles of sklearn estimators |
| numpy | 1.26.4 | same |
| joblib | 1.5.3 | same |

A pickle loaded under another scikit-learn version can refuse to load, or load and predict
differently. Keep these three exact until the models are retrained. `rdkit` and `descriptastorus` are
pinned exactly too: the H237 featurizer must reproduce the 200 `DS_` descriptor columns that both
the RF and the chemprop models were trained on.

### 2.2 The `chemprop` environment is separate and required

```bash
conda create -n chemprop python=3.11.15
conda run -n chemprop pip install -r webapp/requirements-chemprop.txt
```

8 of the 9 endpoints take their **value** from a chemprop model. `app.py` does not import chemprop;
it runs the chemprop CLI as a subprocess, at the path in `CHEMPROP_TRANSFER.chemprop_bin`. If the
binary is missing, the prediction stream ends with an `{"error": ...}` line. It does not fall back
to RF.

The vault holds chemprop **2.2** checkpoints, so keep chemprop on 2.2.x. The training box runs
torch 2.13.0+cu126 (CUDA 12.6). If the server has no GPU, install the CPU build of the same torch
version, then run the end-to-end test in section 10 to prove that the checkpoints predict on CPU.

### 2.3 The shared helpers (Scripts)

`app.py` imports `ML_Reg` from the folder in `webapp.scripts_dir` (default `~/Scripts`). Clone the
whole repo there:

```bash
git clone git@github.com:gtamo39/Scripts.git ~/Scripts
```

At import, `ML_Reg` loads `Statistics_tools` from the same folder, which needs `dill` (pinned in
`requirements.txt`). `xgboost` loads when present and is optional. `Rdkit_tools` is not loaded.

### 2.4 The vault

`config/config.yaml`:

```yaml
MLTRAIL_VAULT: '/data2/MLTrail_vault'
```

Both the deploy (training box) and the webapp (server) open the vault from this key. The webapp
never reads MLTrail's own `config/config.yaml` for paths, so you do not need to edit it.

CAUTION: `registry.json` stores **absolute** model paths, for example
`/data2/MLTrail_vault/models/20_v2.joblib`. Mount the vault at exactly `/data2/MLTrail_vault` on the
server. Changing `MLTRAIL_VAULT` alone does not relocate the models.

If the vault holds no `registry.json` — for example the mount is missing — the app stops at startup
with `FileNotFoundError: MLTRAIL_VAULT: no registry.json in ...`. It does not start with an empty
grid.

Vault footprint: **7.8 GB** (models 7.8 GB, training_sets 57 MB, registry.json 32 KB). The app only
opens `training_sets/` for an older H236 model with no bundled calibration. All 9 pinned RF models
are H237 with calibration, so you can leave `training_sets/` out of the mount. That keeps the
archived training structures off the web server.

### 2.5 What the app does NOT need

- `output/` — 11 GB of training artifacts.
- `data/` — the raw assay export.
- `python/` and `vignettes/` — the training pipeline and the notebooks.
- Write access to the vault. The app only reads (`details()`, `joblib.load()`, `resolve_checkpoint()`).

Every chemprop checkpoint resolves inside the vault, for example
`/data2/MLTrail_vault/models/46_v1/model_0/best.pt`, next to its own `deploy_meta.json` (task list,
target index, descriptor columns).

**The minimal deployment:** this repo's `config/` and `webapp/`, the Scripts clone, the MLTrail
package, the two conda envs, and a read-only mount of the vault.

---

## 3. Startup sequence

Everything happens once, in the FastAPI `lifespan` hook:

1. open the vault at `MLTRAIL_VAULT`
2. bind the 9 RF models by the ids pinned in `MLTRAIL_IDS.rf`
3. bind the chemprop models by the ids pinned in `MLTRAIL_IDS.chemprop`, for each endpoint that
   `webapp.value_model` sends to chemprop
4. load the extra models (`webapp.extra_models`, the Px activity column)
5. build the H237 featurizer
6. with `webapp.strict_startup: true`, stop if a required model did not load

### Which model supplies what

- **Value** — from the model that `webapp.value_model` names: chemprop for 8 endpoints, RF for ppb.
- **Confidence** — *always* from the RF model, for every endpoint. A chemprop regression head returns
  no per-row spread, so the confidence comes from RF tree variance.

So the RF model must load for all 9 endpoints, even where chemprop supplies the value.

### Models are bound by pinned id, not by name

```yaml
MLTRAIL_IDS:
  rf:        {solubility: 20, logd: 21, hlm: 22, mlm: 23, rlm: 24,
              caco2: 25, mdck: 26, mdck_inh: 36, ppb: 27}
  chemprop:  {solubility: 32, logd: 31, hlm: 28, mlm: 29, rlm: 30,
              caco2: 33, mdck: 34, mdck_inh: 46, ppb: 35}
```

The app calls `details(<pinned id>)` directly and never scans the vault by name. `chemprop.ppb: 35`
is kept for provenance; ppb takes its value from RF, so the app does not load it.

### Strict startup

`webapp.strict_startup: true` (the default in this repo) makes a missing model a **startup failure**:

```
RuntimeError: strict_startup: no model for ['mdck_inh (chemprop)']; see the WARN lines above
```

uvicorn then exits, and systemd marks the service failed (section 7). Without it, the app would start
and serve the RF value in silence wherever a chemprop model is missing.

### Auditing the pins — on the training machine

```bash
conda run -n ML python python/ADME_build_ML.py --check_mltrail_ids
# > MLTRAIL_IDS: OK      (exit 0 on 2026-10-19)
```

It checks that each pinned id exists and carries the expected `experiment_name`. Run it on the
**training machine** before you push a config change. It needs the training pipeline, which loads
the CDD Vault client at import, so it does not belong on the web server. On the server, strict
startup is the check.

### Restart after every model redeploy

Model binding runs once. A model registered while the service is up is invisible until restart. A
redeploy appends a new **version** to the same id, so after a restart the app serves the newest
version with no config change. Only a brand-new endpoint needs a new pin.

---

## 4. HTTP surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | HTML shell; stamps `app.js` with its mtime so a browser never runs a stale copy |
| GET | `/static/*` | `app.js`, `index.html` |
| GET | `/api/models` | roster metadata: units, cutoffs, which model gives value and confidence. No structures. |
| POST | `/api/predict` | multipart upload of `.sdf`/`.csv` files; returns an **NDJSON stream** |
| POST | `/api/download_sdf` | `{"run": <run_id>, "rows": [{"idx": ...}]}` → those rows of that run as SDF |

There is no `GET /api/download`. The CSV export is built in the browser from the grid it already
holds.

### `/api/predict` is a stream

The body is newline-delimited JSON: one `{"label","done","total"}` line per finished stage, then one
final `{"result": {"rows", "n", "n_valid", "run_id"}}` line, or `{"error": "..."}` if a stage fails.
A run takes tens of seconds, because each chemprop endpoint is its own CLI subprocess.

The route sets the headers that keep a proxy from buffering the stream:

```
Cache-Control: no-cache, no-transform
X-Accel-Buffering: no
```

**Your reverse proxy must not buffer this response** (nginx: `proxy_buffering off;`). If it buffers,
the page shows no progress and then everything at once. Also set `proxy_read_timeout` well above the
longest run.

---

## 5. Config you set on the server

All in `config/config.yaml`:

| key | value here | what it does |
|---|---|---|
| `webapp.host` | `127.0.0.1` | bind address. Keep it; put the proxy on the same host |
| `webapp.port` | `8050` | |
| `webapp.scripts_dir` | `~/Scripts` | folder of the Scripts clone |
| `webapp.strict_startup` | `true` | stop when a required model does not load |
| `webapp.run_ttl_min` | `60` | minutes a scored run stays available to the SDF export |
| `webapp.max_runs` | `50` | scored runs held in memory at most; the oldest go first |
| `MLTRAIL_VAULT` | `/data2/MLTrail_vault` | the vault; must match the mount path (section 2.4) |
| `CHEMPROP_TRANSFER.chemprop_bin` | `~/miniconda3/envs/chemprop/bin/chemprop` | the chemprop CLI |

**Keep the config identical to git.** `~` expands against the account that runs the service. So if
the service account has `~/miniconda3` and `~/Scripts`, the repo config works unchanged and
`git pull` never conflicts. A local edit on the server drifts from the training box.

Everything else under `webapp:` is presentation and science (`endpoint_order`, `cutoffs`,
`value_model`, `color_slope`, `confidence_split`, `extra_models`). Do not tune those to fix a
deployment problem; coordinate with the repo owner.

---

## 6. Behaviour with more than one user

### 6.1 Downloads are tied to a random run id

Each `/api/predict` stores its result under `secrets.token_urlsafe(16)` (22 characters) and returns
that id in its final line. The browser keeps it and sends it with the SDF export. The export serves
**only** the rows of the run that the id names. An unknown, absent or expired id gets HTTP 404.

The store is in memory. It drops runs older than `run_ttl_min` and keeps at most `max_runs`. A
restart clears it.

Remaining exposure: the id is a bearer token. Whoever holds it can export that run until it expires.
It travels only in the response to the uploader's browser. This does not replace authentication:
anyone who reaches the app can still upload their own files (section 6.3).

Verified 2026-10-19 against the live vault: the right id returned 200 with 3 records, a made-up id
returned 404, and no id returned 404. Unit tests: `python -m unittest tests.test_webapp`.

### 6.2 Run exactly one worker

The run store lives in one process. With `--workers > 1`, an export routed to a worker that did not
score the file returns 404. That fails safe, but it fails. Run one worker. Horizontal scaling needs a
shared store first.

### 6.3 There is no authentication, authorization or rate limiting

Authentication must come from the reverse proxy: company SSO, or at minimum mTLS or HTTP auth on a
restricted network. Do not expose the port directly.

Uploads are read fully into memory with no size cap. Set `client_max_body_size` on the proxy.

### 6.4 One prediction at a time, roughly

A run holds a core for tens of seconds: 9 RF models scored over all trees, plus one chemprop
subprocess per chemprop endpoint. Requests queue. Size the host for the number of *simultaneous*
chemists, and set generous proxy timeouts.

---

## 7. Running it as a service

### Manual start, for a first smoke test

```bash
cd ~/ADME_ML
conda run -n ML python webapp/app.py        # reads webapp.host / webapp.port
```

### systemd unit

The paths assume a service account `adme` whose home holds the repo, the Scripts clone and
miniconda, so the shared config works unchanged (section 5).

```ini
[Unit]
Description=ADME property prediction webapp
After=network.target
# stop retrying after 3 failed starts in 5 minutes, so a bad pin shows as "failed"
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=adme
Group=adme
WorkingDirectory=/home/adme/ADME_ML

# call the interpreter directly; one worker (section 6.2)
ExecStart=/home/adme/miniconda3/envs/ML/bin/python -m uvicorn webapp.app:app \
          --host 127.0.0.1 --port 8050 --workers 1 --timeout-keep-alive 120

Environment=PYTHONUNBUFFERED=1
Environment=MPLBACKEND=Agg

Restart=on-failure
RestartSec=5

# the app writes only to its own temp folders
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
ReadOnlyPaths=/data2/MLTrail_vault
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Notes:

- `StartLimit*` matters because of strict startup. Without it, a config error restarts the service
  every 5 seconds forever.
- `MPLBACKEND=Agg` — `ML_Reg` imports `matplotlib.pyplot` at import. A headless backend keeps that
  import from looking for a display.
- `conda run` is not used in `ExecStart`, because it complicates signal handling on stop and restart.
- `ProtectSystem=strict` and `ProtectHome=read-only` make the whole filesystem read-only. The chemprop
  subprocess wrote nothing to the working directory during the verification run. Writes to a home
  cache (`~/.cache`) are not proven either way. Run the section 10 test under the unit. If the
  journal shows a write error, add `ReadWritePaths=` for that one path.
- `PrivateTmp=true` gives the service its own `/tmp`.
- `ReadOnlyPaths` encodes the fact that the app never writes to the vault. Keep it.

---

## 8. Reverse proxy sketch (nginx)

```nginx
location / {
    # company SSO / auth goes here — the app itself has none
    auth_request /oauth2/auth;

    proxy_pass http://127.0.0.1:8050;
    proxy_http_version 1.1;

    # REQUIRED: /api/predict is a progress stream (section 4)
    proxy_buffering off;
    proxy_cache off;

    proxy_read_timeout 600s;       # a large upload can run for minutes
    client_max_body_size 32m;      # uploads are held in memory; cap them
}
```

The page loads three web fonts from `fonts.googleapis.com`. The **browser** fetches them, not the
server, so the server needs no outbound internet for them. The requests carry no chemistry data.

---

## 9. Operational notes

### 9.1 Restart after every model redeploy

Section 3. It is the most common support question.

### 9.2 chemprop temp folders

Each chemprop prediction runs in a `tempfile.TemporaryDirectory`, which Python removes after the
call. Verified 2026-10-19: 8 chemprop predictions left the count of `/tmp/cp_pred_*` folders
unchanged.

The shared `ML_Reg.chemprop_predict_from_checkpoint` still leaves its folder behind when a caller
gives it no `workdir`. Other projects call it that way; the webapp no longer does.

### 9.3 Upload staging

Uploads land in `tempfile.mkdtemp(prefix='adme_webapp_')`. Each file is deleted right after parsing,
and the folder is removed at exit. A `SIGKILL` skips the exit hook; `PrivateTmp=true` covers that.

### 9.4 Logs

The app prints to stdout; under systemd that goes to the journal. A healthy start prints:

```
> loaded 9 RF models: [...]
> loaded 8 chemprop VALUE models: [...]
> confidence comes from RF for all 9 endpoints
```

and no `WARN` line. Alert on:

```
WARN: <ep>: ...                                  # a pinned model did not load
RuntimeError: strict_startup: no model for [...]  # the service refused to start
FileNotFoundError: MLTRAIL_VAULT: ...             # the vault is not mounted
```

`/api/predict` reports a failure **inside the 200 OK stream**, as a final `{"error": ...}` line.
Monitoring that only watches status codes sees a healthy service while every prediction fails.

---

## 10. Verification checklist

Run these on the server as the service account, from the repo root.

```bash
PY=~/miniconda3/envs/ML/bin/python

# 1. the vault is mounted at the configured path
ls /data2/MLTrail_vault/registry.json

# 2. the chemprop CLI runs in its own env
~/miniconda3/envs/chemprop/bin/chemprop --help | head -3

# 3. the shared helpers import from the Scripts clone
$PY -c "import sys, os; sys.path.insert(0, os.path.expanduser('~/Scripts')); import ML_Reg; print('ML_Reg OK')"

# 4. the featurizer builds (needs descriptastorus)
$PY -c "import yaml; from mltrail.featurizers import get_featurizer; \
  get_featurizer('H237', yaml.safe_load(open('config/config.yaml'))); print('H237 OK')"

# 5. the app module imports, and binds every model
$PY -c "
import sys; sys.path.insert(0, '.')
from webapp import app as A
A._discover_champions(); A._discover_cp_values()
print(f'RF {len(A.CHAMPIONS)}/9, chemprop {len(A.CP_VALUES)}, missing {A._missing_models()}')"

# 6. the unit tests for the run store and the startup check
$PY -m unittest tests.test_webapp
```

Step 5 must print `RF 9/9, chemprop 8, missing []`.

### End-to-end test, with public molecules only

Put this in a script file and run it with `bash`. Pasted as one line, the steps can chain wrongly and
leave a stale server answering.

```bash
PY=~/miniconda3/envs/ML/bin/python; PORT=8078
printf 'compound,smiles\npub_001,CCO\npub_002,c1ccccc1\npub_003,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n' > /tmp/public3.csv

# start, wait for the models
$PY -m uvicorn webapp.app:app --host 127.0.0.1 --port $PORT --workers 1 > /tmp/adme_e2e.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -sf -o /dev/null localhost:$PORT/api/models && break; sleep 3; done

# score, then export with the run id
RUN=$(curl -s -F "files=@/tmp/public3.csv" localhost:$PORT/api/predict | tail -1 | \
      $PY -c "import json,sys; print(json.load(sys.stdin)['result']['run_id'])")
curl -s -o /dev/null -w 'right id: %{http_code}\n' -H 'Content-Type: application/json' \
     -d "{\"run\":\"$RUN\",\"rows\":[{\"idx\":0},{\"idx\":1},{\"idx\":2}]}" localhost:$PORT/api/download_sdf
curl -s -o /dev/null -w 'made-up id: %{http_code}\n' -H 'Content-Type: application/json' \
     -d '{"run":"made-up","rows":[{"idx":0}]}' localhost:$PORT/api/download_sdf

# stop this server only
kill $SRV
```

Expected: `right id: 200`, `made-up id: 404`. Result on the training box on 2026-10-19: 3 of 3
compounds scored, 200 with 3 SDF records, 404 for a made-up id and for no id.

---

## 11. Open questions for the repo owner

Resolved on 2026-10-19:

- Vault path: the exact current path, `/data2/MLTrail_vault`, now in `MLTRAIL_VAULT`.
- Vault mount: read-only.
- Download isolation: done (section 6.1).
- Internet for the fonts: not needed; the browser fetches them (section 8).

Still open:

1. **Who restarts the service after a model redeploy?** Retraining and restarting are coupled
   (section 3).
2. **Reuse the existing service setup?** A FastAPI service already runs on this network:
   `serac-chemistry-services`, uvicorn on `0.0.0.0:8001`, owned by `fchevillard`. Its unit file,
   proxy config and auth layer are a better starting point than the generic sketch in section 7.
   Port 8050 does not collide with 8001.

---

## 12. Remaining work before go-live

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Per-run download ids (section 6.1) | Repo | **done** 2026-10-19 |
| 2 | Vault path as config key `MLTRAIL_VAULT` (section 2.4) | Repo | **done** 2026-10-19 |
| 3 | Scripts path as config key `webapp.scripts_dir` (section 2.3) | Repo | **done** 2026-10-19 |
| 4 | Remove the chemprop temp folder after each call (section 9.2) | Repo | **done** 2026-10-19 |
| 5 | Strict startup (section 3) | Repo | **done** 2026-10-19 |
| 6 | Pin both envs: `webapp/requirements*.txt` (section 2) | Repo | **done** 2026-10-19 |
| 7 | Commit and push these changes | Repo owner | open — the server deploys from git |
| 8 | Answer the two open questions (section 11) | Repo owner | open |
| 9 | Service account with miniconda, both envs, and the Scripts + MLTrail clones (section 2) | Server | open |
| 10 | Read-only mount of `/data2/MLTrail_vault` at that exact path (section 2.4) | Server | open |
| 11 | systemd unit, one worker, start limit (section 7) | Server | open |
| 12 | Reverse proxy: SSO, TLS, no buffering, timeouts, body size cap (section 8) | Server | open |
| 13 | Run section 10 on the server, including the end-to-end test | Server | open |
| 14 | Monitoring that reads the stream's `{"error": ...}` line (section 9.4) | Server | open |
