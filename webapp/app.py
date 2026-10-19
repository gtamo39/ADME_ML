#!/usr/bin/env python3
"""app.py — FastAPI server for drag-drop ADME property prediction.

This app handles SMILES / compound ids / predicted values. It binds config webapp.host
(127.0.0.1 by default). On a shared server, reach it only through an authenticating reverse
proxy on the same host — see docs/handoff.md.

Pipeline (reusing the MLTrail vault + the ADME config):
  startup -> open the vault at config MLTRAIL_VAULT
          -> bind each RF model by the id pinned in config MLTRAIL_IDS.rf (every endpoint)
          -> load each estimator + trained feature columns + conf_recal calibration (from the bundle)
          -> fallback scale = std of the archived training-set labels when no calibration is bundled
          -> bind the chemprop value models (MLTRAIL_IDS.chemprop) for the endpoints that config
             `webapp.value_model` sends to chemprop; RF still loads for every endpoint
          -> load any config `webapp.extra_models` (non-ADME MLTrail models, own grid column)
          -> with config webapp.strict_startup, stop when a required model did not load
  upload  -> read the dropped SDF/CSV once -> featurize H237 once
          -> per model: mean (raw value) + per-tree std (-> conf_recal confidence) -> favorable flag
          -> chemprop endpoints: REPLACE the value with the chemprop prediction, KEEP the RF confidence
          -> per extra classifier: P(positive class) + the decision margin |2p-1| as the confidence
          -> keep the run under a random run id; the SDF export must send that id back
  render  -> LiveDesign-style grid, one diagonal-split cell per endpoint

Predictions/SMILES/ids go only to the browser that uploaded them; nothing crosses to any cloud.
"""
import atexit
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from mltrail import Registry
from mltrail.config import default_config
from mltrail.backends import align_features, resolve_checkpoint
from mltrail.featurizers import get_featurizer
from mltrail.readers import read_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
WEBAPP = CONFIG.get("webapp", {})
ENDPOINT_CFG = CONFIG["ADME_ENDPOINTS"]                 # transform + unit per endpoint

# shared helpers from the configured Scripts checkout
sys.path.insert(0, os.path.expanduser(WEBAPP.get("scripts_dir", "~/Scripts")))
import ML_Reg

# Favorable/unfavorable cell colors (olive / red); confidence uses SERAC azure/ember (config).
PALETTE = {
    "azure": CONFIG["SERAC_C"]["azure"], "ember": CONFIG["SERAC_C"]["ember"],
    "fav": CONFIG["SERAC_C"].get("olive", "#74B24A"), "unfav": "#C94C3C",
    "split": WEBAPP.get("confidence_split", 0.5),
}

# Steepness of the favorable<->unfavorable value fade, per MODELLING-space unit from the cutoff.
# 2.0 means one log unit (or one logD unit) past the cutoff reaches ~88% of the full color.
COLOR_SLOPE = float(WEBAPP.get("color_slope", 2.0))

# raw-unit -> modelling-space inverse transforms (mirror ADME_build_ML._TF inverses).
_INV = {
    "log10": lambda a: np.power(10.0, a),
    "identity": lambda a: a,
    "logit_pct": lambda a: 100.0 / (1.0 + np.power(10.0, -a)),   # logit(%unbound) -> % unbound
}

# Temp staging for dropped files — cleared on exit; never written into the repo.
STAGE_DIR = Path(tempfile.mkdtemp(prefix="adme_webapp_"))
atexit.register(lambda: shutil.rmtree(STAGE_DIR, ignore_errors=True))

VALUE_MODEL = WEBAPP.get("value_model", {})   # endpoint -> 'rf' | 'chemprop' (which model gives the VALUE)
MLTRAIL_IDS = CONFIG.get("MLTRAIL_IDS", {})   # framework -> endpoint -> pinned vault id (None = not deployed)

CHAMPIONS = {}          # endpoint -> {model, feature_cols, sigma, transform, unit, cutoff, favorable, model_id}
CP_VALUES = {}          # endpoint -> {checkpoint, tasks, descriptor_cols, transform, model_id} (value only)
EXTRAS = {}             # key -> {model, feature_cols, pos_idx, label, unit, threshold, model_id} (extra columns)
_FEATURIZE = None       # H237 featurizer (built once at startup)
RUNS = {}               # run id -> {flat, blocks, t}: the scored runs the SDF export can ask for
RUNS_LOCK = threading.Lock()


# ---------- startup: discover + load the champions ----------

def _registry():
    """Open the MLTrail vault at config MLTRAIL_VAULT; raise FileNotFoundError when it holds no
    registry.json."""
    vault = Path(CONFIG["MLTRAIL_VAULT"])
    # fail on a vault that is not mounted
    if not (vault / "registry.json").exists():
        raise FileNotFoundError(f"MLTRAIL_VAULT: no registry.json in {vault}")

    # MLTrail defaults, with the three vault paths from this config
    return Registry.from_config({**default_config(), "registry_path": str(vault / "registry.json"),
                                 "trained_models_dir": str(vault / "models"),
                                 "training_sets_dir": str(vault / "training_sets")})


def _discover_champions():
    """Load the deployed RF model for every endpoint, by the id pinned in config MLTRAIL_IDS.rf.
    The pin is authoritative: the vault listing is never scanned and its order never matters. Loads
    model + feature_cols + calibration from each bundle; an H236 entry has no conf_recal, so its
    confidence scale falls back to the std of the archived training labels. These models supply the
    CONFIDENCE for every endpoint, so an endpoint with no usable pin is dropped from the app."""
    reg = _registry()
    cutoffs = WEBAPP.get("cutoffs", {})
    # bind the id pinned in config MLTRAIL_IDS.rf; an unpinned or missing entry drops the endpoint
    for ep in ENDPOINT_CFG:
        mid = MLTRAIL_IDS.get("rf", {}).get(ep)
        if mid is None:
            print(f"WARN: {ep}: no MLTRAIL_IDS.rf pin; endpoint dropped (it has no confidence)", flush=True)
            continue
        try:
            d = reg.details(int(mid))
        except Exception as ex:
            print(f"WARN: {ep}: MLTRAIL_IDS.rf id {mid} unavailable ({ex}); endpoint dropped", flush=True)
            continue
        ft = d.get("features_type")
        # the bundle carries model + trained columns + (for H237) the conf_recal calibration
        bundle = joblib.load(d["model_path"])
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else None
        calib = bundle.get("calibration") if isinstance(bundle, dict) else None
        # fallback scale (training-label std) only when no conf_recal calibration is bundled
        sigma = None
        if calib is None:
            try:
                sigma = float(reg.load_training_set(d["id"])["label"].std()) or 1.0
            except Exception as ex:
                print(f"WARN: {ep}: training-label std unavailable ({ex}); sigma=1.0", flush=True)
                sigma = 1.0
        cut = cutoffs.get(ep, {})
        CHAMPIONS[ep] = {
            "model": model, "feature_cols": feature_cols, "features_type": ft,
            "calibration": calib, "sigma": sigma,
            "transform": ENDPOINT_CFG[ep]["transform"], "unit": ENDPOINT_CFG[ep].get("unit", ""),
            "cutoff": cut.get("value"), "favorable": cut.get("favorable"), "model_id": int(d["id"]),
        }
    fts = sorted(f"{ep}:{CHAMPIONS[ep]['features_type']}" for ep in CHAMPIONS)
    print(f"> loaded {len(CHAMPIONS)} RF models: {fts}", flush=True)


def _discover_cp_values():
    """Load the deployed chemprop model for every endpoint that config `webapp.value_model` sends to
    chemprop, by the id pinned in config MLTRAIL_IDS.chemprop. These supply the VALUE only — the confidence keeps coming
    from the RF champion, because a chemprop regression head returns no per-row std. Each model
    directory describes itself in `deploy_meta.json` (task list, target index, DS_ descriptor columns),
    written by OUTPUT.deploy_endpoint_cp; MLTrail's own chemprop_predict is not used, because it sends
    SMILES only and these models need the descriptor columns."""
    want = {ep for ep, m in VALUE_MODEL.items() if m == "chemprop"}
    bad = {ep for ep, m in VALUE_MODEL.items() if m not in ("rf", "chemprop")}
    if bad:
        print(f"WARN: webapp.value_model has unsupported model(s) {sorted(bad)}; those endpoints use RF", flush=True)
    if not want:
        return
    reg, cp_bin = _registry(), CONFIG["CHEMPROP_TRANSFER"]["chemprop_bin"]
    for ep in sorted(want):
        mid = MLTRAIL_IDS.get("chemprop", {}).get(ep)
        if mid is None:
            print(f"WARN: {ep}: value_model says chemprop but MLTRAIL_IDS.chemprop has no pin; "
                  f"falling back to the RF value", flush=True)
            continue
        try:
            d = reg.details(int(mid))
        except Exception as ex:
            print(f"WARN: {ep}: MLTRAIL_IDS.chemprop id {mid} unavailable ({ex}); using the RF value", flush=True)
            continue
        meta_path = Path(d["model_path"]) / "deploy_meta.json"
        if not meta_path.exists():
            print(f"WARN: {ep}: {meta_path} missing; falling back to the RF value", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        CP_VALUES[ep] = {
            "checkpoint": str(resolve_checkpoint(d["model_path"])), "chemprop_bin": os.path.expanduser(cp_bin),
            "tasks": meta["tasks"], "descriptor_cols": meta["descriptor_cols"],
            "transform": meta["transform"], "grouping": meta["grouping"], "model_id": int(d["id"]),
        }
    missing = sorted(want - set(CP_VALUES))
    if missing:
        print(f"WARN: no deployed adme_<ep>_cp for {missing}; those endpoints use the RF value", flush=True)
    got = ", ".join(f"{ep}:{c['grouping']}(id {c['model_id']})" for ep, c in sorted(CP_VALUES.items()))
    print(f"> loaded {len(CP_VALUES)} chemprop VALUE models: {got}", flush=True)
    print(f"> confidence comes from RF for all {len(CHAMPIONS)} endpoints", flush=True)


def _load_extras():
    """Load the config `webapp.extra_models` — MLTrail models that are NOT ADME endpoints (e.g. the
    Px single/low activity classifier, id 19). They get their own grid column and their own CSV columns,
    but they stay OUT of the MPO formula scope. Classification only: the column shows P(positive class)."""
    reg = _registry()
    for spec in WEBAPP.get("extra_models", []):
        mid = int(spec["model_id"])
        try:
            d = reg.details(mid)
            bundle = joblib.load(d["model_path"])
        except Exception as ex:                      # a missing vault entry must not stop the ADME app
            print(f"WARN: extra model {mid} ({spec['key']}) unavailable ({ex}); column skipped", flush=True)
            continue
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        EXTRAS[spec["key"]] = {
            "model": model, "feature_cols": bundle.get("feature_cols") if isinstance(bundle, dict) else None,
            "pos_idx": int(np.argmax(model.classes_)),      # column of the positive (highest) class label
            "label": spec.get("label", spec["key"]), "unit": spec.get("unit", ""),
            "threshold": float(spec.get("threshold", 0.5)),
            "color_slope": float(spec.get("color_slope", 8.0)),
            "features_type": d.get("features_type"), "model_id": mid,
        }
        print(f"> extra model {mid}: {d.get('experiment_name')} ({d.get('model_type')}, "
              f"{d.get('features_type')}) -> column '{spec['key']}'", flush=True)


def _ordered_endpoints():
    """Endpoints in the configured column order, restricted to the champions actually loaded."""
    order = WEBAPP.get("endpoint_order", sorted(CHAMPIONS))
    return [e for e in order if e in CHAMPIONS] + [e for e in CHAMPIONS if e not in order]


# ---------- helpers ----------

def _svg(smiles, width=150, height=100, bond_line_width=None):
    """Render a compound to an inline SVG string (local, RDKit). Empty string on parse failure.
    bond_line_width: absolute (unscaled) bond stroke — set thin for the crisp high-res hover preview."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    if bond_line_width is not None:
        opts.bondLineWidth = bond_line_width      # absolute px width...
        opts.scaleBondWidth = False               # ...kept constant so a big canvas => thin, clean lines
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _confidence(std, c):
    """Per-prediction confidence in (0,1] from the tree-variance std (modelling space).
    conf_recal when the model bundles a calibration: exp(-clip(recal_a + recal_b*std, 0)/rmse_cv);
    else the training-label-std fallback exp(-std/sigma)."""
    calib = c.get("calibration")
    if calib and np.isfinite(calib.get("rmse_cv", np.nan)) and calib["rmse_cv"] > 0:
        resid = np.clip(calib.get("recal_a", 0.0) + calib.get("recal_b", 0.0) * std, 0.0, None)
        return np.exp(-resid / calib["rmse_cv"])
    return np.exp(-std / (c.get("sigma") or 1.0))


def _is_favorable(raw, cutoff, sign):
    """True if a raw predicted value passes the triage cutoff in the favorable direction."""
    if cutoff is None or sign is None or raw is None or not np.isfinite(raw):
        return None
    return raw >= cutoff if sign == ">=" else (raw <= cutoff if sign == "<=" else
           raw > cutoff if sign == ">" else raw < cutoff)


def _read_sdf_keep_blocks(path):
    """Read an SDF into df[compound, smiles] AND the untouched text of each record it kept.

    mltrail's reader SKIPS records RDKit cannot parse, so the surviving rows no longer line up with
    the file's record order. Re-splitting the file separately would therefore attach a prediction to
    the wrong molecule. This parses and splits in ONE pass, so the returned blocks are aligned to the
    returned rows by construction. Each block is the original text verbatim, up to and including
    'M  END' plus any data fields — the export re-emits it byte for byte.
    """
    text = Path(path).read_text(errors="replace")
    ids, smiles, blocks = [], [], []
    for i, record in enumerate(text.split("$$$$")):
        # every record after the first inherits the separator's newline; a leading blank line shifts
        # the molblock header (title/program/comment/counts) and RDKit then fails to parse it
        record = record.lstrip("\n")
        if not record.strip():
            continue
        mol = Chem.MolFromMolBlock(record)              # same parse+sanitize as SDMolSupplier
        if mol is None:
            continue                                    # mirrors mltrail's skip rule
        ids.append(mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else i)
        smiles.append(Chem.MolToSmiles(mol))
        blocks.append(record)
    return pd.DataFrame({"compound": ids, "smiles": smiles}), blocks


def _read_upload(path, filename):
    """Read one dropped file into (df[compound, smiles], sdf_blocks). blocks is None for a csv:
    there is no original record to preserve, so the SDF export builds one from the SMILES."""
    ext = Path(filename).suffix.lower()
    if ext == ".sdf":
        return _read_sdf_keep_blocks(path)
    head = pd.read_csv(path, nrows=0)
    cols = list(head.columns)
    smi_col = next((c for c in cols if "smiles" in c.lower()), None)
    if smi_col is None:
        raise HTTPException(400, f"{filename}: no SMILES column found (looked for a name containing 'smiles')")
    id_col = next((c for c in cols if any(k in c.lower() for k in ("compound", "_id", "id", "name"))), None)
    return read_dataset(path, smiles_column=smi_col, compound_id=id_col), None


def _predict_all(df):
    """Drain _predict_iter and return (rows, flat_df) — for a caller that wants only the answer."""
    for item in _predict_iter(df):
        if item[0] == "done":
            return item[1], item[2]
    raise RuntimeError("_predict_iter ended without a result")


def _predict_iter(df):
    """Featurize H237 once, score every champion, and YIELD progress as each model finishes.

    The VALUE comes from the model config `webapp.value_model` names (RF, or chemprop where it wins);
    the CONFIDENCE always comes from the RF champion's per-tree std.

    Yields ("step", label, done, total) after every stage, then ("done", rows, flat_df) exactly once.
    The route turns those into an NDJSON stream, so the browser can draw a real progress bar instead
    of guessing — the chemprop stages are seconds each, because every one is a CLI subprocess.

    rows: per-compound dicts for the grid (compound, smiles, svg, valid, preds{ep:{value,favorable,confidence}}).
    flat_df: the same data flattened to one row per compound for CSV download.
    """
    # stages: featurize + one per RF model + one per chemprop override + one per extra + the render
    total = 2 + len(CHAMPIONS) + len(CP_VALUES) + len(EXTRAS)
    done = 0

    df = df.reset_index(drop=True)
    df["_key"] = np.arange(len(df))
    feats = _FEATURIZE(pd.DataFrame({"compound": df["_key"].values, "smiles": df["smiles"].values}))
    keys = feats["compound"].to_numpy()                     # _key of each row that featurized (bad SMILES dropped)
    pos = {k: i for i, k in enumerate(keys)}
    done += 1
    yield "step", "featurizing", done, total

    # per endpoint: mean (raw) + confidence for every valid row, indexed by _key
    ep_out = {}
    for ep, c in CHAMPIONS.items():
        X = align_features(feats, c["feature_cols"])
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)   # drop names: trees were fitted on a plain array
        per_tree = np.stack([est.predict(Xv) for est in c["model"].estimators_])   # (n_trees, n_valid)
        raw = _INV[c["transform"]](per_tree.mean(axis=0))
        conf = _confidence(per_tree.std(axis=0), c)                                # (0, 1], conf_recal or fallback
        ep_out[ep] = (raw, conf)
        done += 1
        yield "step", f"{ep} (RF)", done, total

    # chemprop VALUE override: replace the value for its endpoints, KEEP the RF confidence (index 1).
    # chemprop needs the SMILES plus the same DS_ descriptor columns it trained on, so it gets the
    # featurized frame with the smiles of the rows that survived featurization, in feats row order.
    if CP_VALUES:
        cp_in = feats.assign(smiles=df.set_index("_key").loc[keys, "smiles"].to_numpy())
        for ep, c in CP_VALUES.items():
            if ep not in ep_out:
                continue
            # score in a temp workdir that is removed after the call
            with tempfile.TemporaryDirectory(prefix="cp_pred_") as wd:
                pred = ML_Reg.chemprop_predict_from_checkpoint(
                    cp_in, "compound", c["checkpoint"], c["chemprop_bin"], target=ep, targets=c["tasks"],
                    descriptor_cols=c["descriptor_cols"], workdir=wd)
            ep_out[ep] = (_INV[c["transform"]](pred["pred_y"].to_numpy(float)), ep_out[ep][1])
            done += 1
            yield "step", f"{ep} (chemprop)", done, total

    # extra classifiers: value = P(positive class); confidence = decision margin |2p-1| (0 at the
    # 0.5 boundary, 1 at a unanimous forest). NOTE: a per-tree std is useless here — the trees vote 0/1,
    # so std ~ sqrt(p(1-p)) and carries no information beyond p itself.
    for xk, c in EXTRAS.items():
        X = align_features(feats, c["feature_cols"])
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        prob = c["model"].predict_proba(Xv)[:, c["pos_idx"]]
        ep_out[xk] = (prob, np.abs(2.0 * prob - 1.0))
        done += 1
        yield "step", c["label"], done, total

    order = _ordered_endpoints()
    rows, flat = [], []
    for _, r in df.iterrows():
        key, smi = int(r["_key"]), str(r["smiles"])
        valid = key in pos
        preds, flat_row = {}, {"compound": r["compound"], "smiles": smi}
        for ep in order:
            c = CHAMPIONS[ep]
            if valid:
                raw = float(ep_out[ep][0][pos[key]]); conf = float(ep_out[ep][1][pos[key]])
                fav = _is_favorable(raw, c["cutoff"], c["favorable"])
                preds[ep] = {"value": raw, "favorable": fav, "confidence": conf}
                flat_row[f"{ep}_pred"] = raw; flat_row[f"{ep}_confidence"] = conf
            else:
                preds[ep] = {"value": None, "favorable": None, "confidence": None}
                flat_row[f"{ep}_pred"] = None; flat_row[f"{ep}_confidence"] = None
        # extra (non-ADME) columns — same cell shape, but the value is a class probability
        for xk, c in EXTRAS.items():
            if valid:
                prob = float(ep_out[xk][0][pos[key]]); conf = float(ep_out[xk][1][pos[key]])
                preds[xk] = {"value": prob, "favorable": bool(prob >= c["threshold"]), "confidence": conf}
                flat_row[f"{xk}_pred"] = prob; flat_row[f"{xk}_confidence"] = conf
            else:
                preds[xk] = {"value": None, "favorable": None, "confidence": None}
                flat_row[f"{xk}_pred"] = None; flat_row[f"{xk}_confidence"] = None
        # grid thumbnail (small) + a fresh high-resolution render for the hover preview (thin, clean bonds)
        svg = _svg(smi) if valid else ""
        svg_hi = _svg(smi, width=400, height=300, bond_line_width=1.2) if valid else ""
        rows.append({"compound": r["compound"], "smiles": smi, "svg": svg, "svg_hi": svg_hi, "valid": valid, "preds": preds})
        flat.append(flat_row)
    yield "step", "drawing structures", total, total
    yield "done", rows, pd.DataFrame(flat)


def _store_run(flat, blocks):
    """Keep one scored run under a new random id, for the SDF export. Drops the runs older than
    config webapp.run_ttl_min, then the oldest beyond webapp.max_runs.
    param DataFrame flat: one row per compound (compound, smiles, <ep>_pred, <ep>_confidence)
    param list blocks: the original SDF record per row; None for a row from a csv
    return str: the run id the client must send back
    """
    run_id, now = secrets.token_urlsafe(16), time.time()
    ttl, cap = 60 * float(WEBAPP.get("run_ttl_min", 60)), int(WEBAPP.get("max_runs", 50))
    with RUNS_LOCK:
        # drop expired runs
        for k in [k for k, r in RUNS.items() if now - r["t"] > ttl]:
            del RUNS[k]

        # drop the oldest runs beyond the cap
        for k in list(RUNS)[:max(0, len(RUNS) + 1 - cap)]:
            del RUNS[k]

        # keep this run
        RUNS[run_id] = {"flat": flat, "blocks": blocks, "t": now}
    return run_id


def _get_run(run_id):
    """Return the stored run for run_id; raise HTTP 404 when the id is unknown or expired."""
    ttl = 60 * float(WEBAPP.get("run_ttl_min", 60))
    with RUNS_LOCK:
        run = RUNS.get(run_id or "")
    if run is None or time.time() - run["t"] > ttl:
        raise HTTPException(404, "unknown or expired run; drop the file again to score it")
    return run


def _missing_models():
    """Required models that did not load: the RF model of every endpoint, plus the chemprop model
    of every endpoint that config webapp.value_model sends to chemprop."""
    return ([ep for ep in ENDPOINT_CFG if ep not in CHAMPIONS] +
            [f"{ep} (chemprop)" for ep, m in VALUE_MODEL.items() if m == "chemprop" and ep not in CP_VALUES])


# ---------- app ----------

@asynccontextmanager
async def lifespan(app):
    global _FEATURIZE
    _discover_champions()                 # RF for ALL endpoints — it always supplies the confidence
    _discover_cp_values()                 # chemprop models that override the VALUE (config webapp.value_model)
    _load_extras()                        # non-ADME extra-column models (config webapp.extra_models)
    _FEATURIZE = get_featurizer("H237", CONFIG)   # H237 featurizer (H237 columns cover H236 fallbacks)

    # stop the service when a required model did not load
    missing = _missing_models()
    if missing and WEBAPP.get("strict_startup", False):
        raise RuntimeError(f"strict_startup: no model for {missing}; see the WARN lines above")
    yield


app = FastAPI(title="ADME property prediction", lifespan=lifespan)


@app.get("/")
def index():
    """Serve the shell, stamping app.js with its own mtime.

    The version query used to be hand-edited, so a code change could be — and was — served from the
    browser cache, leaving an old app.js talking to a new API. The mtime makes that impossible.
    """
    html = (STATIC_DIR / "index.html").read_text()
    v = int((STATIC_DIR / "app.js").stat().st_mtime)
    html = re.sub(r'(/static/app\.js)(\?v=[^"\']*)?', r'\1?v=%d' % v, html)
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/api/models")
def models():
    """Champion roster (endpoint, unit, cutoff, favorable direction, which model gives the value and
    which gives the confidence) + the cell palette. Metadata only."""
    order = _ordered_endpoints()
    eps = [{"key": ep, "unit": CHAMPIONS[ep]["unit"], "cutoff": CHAMPIONS[ep]["cutoff"],
            "favorable": CHAMPIONS[ep]["favorable"], "transform": CHAMPIONS[ep]["transform"],
            "color_slope": COLOR_SLOPE, "model_id": CHAMPIONS[ep]["model_id"],
            "value_model": "chemprop" if ep in CP_VALUES else "rf",
            "value_model_id": CP_VALUES[ep]["model_id"] if ep in CP_VALUES else CHAMPIONS[ep]["model_id"],
            "confidence_model_id": CHAMPIONS[ep]["model_id"]} for ep in order]
    # a probability lives on a 0-1 scale, so its fade needs a steeper slope than a log-unit endpoint
    extras = [{"key": k, "label": c["label"], "unit": c["unit"], "cutoff": c["threshold"],
               "favorable": ">=", "kind": "classification", "transform": "identity",
               "color_slope": c["color_slope"], "model_id": c["model_id"]}
              for k, c in EXTRAS.items()]
    return {"endpoints": eps, "extras": extras, "palette": PALETTE}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    """Accept one or more dropped .sdf/.csv files; STREAM per-compound predictions for the grid.

    The body is newline-delimited JSON: one {"label","done","total"} line per finished stage, then a
    final {"result": {...}} line that carries this run's random run_id. Streaming, not one JSON blob, because a run takes tens of seconds —
    every chemprop endpoint is its own CLI subprocess — and the browser draws a real progress bar
    from these lines. An error mid-run arrives as a final {"error": "..."} line.
    """
    frames, blocks = [], []
    for uf in files:
        staged = STAGE_DIR / f"{uuid.uuid4().hex}{Path(uf.filename).suffix.lower()}"
        staged.write_bytes(await uf.read())
        try:
            frame, blk = _read_upload(str(staged), uf.filename)
        finally:
            staged.unlink(missing_ok=True)
        frames.append(frame)
        # a csv contributes no original record; pad so `blocks` stays aligned with the concat
        blocks += blk if blk is not None else [None] * len(frame)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["compound", "smiles"])

    def stream():
        if len(df) == 0:
            yield json.dumps({"result": {"rows": [], "n": 0, "n_valid": 0, "run_id": None}}) + "\n"
            return
        try:
            for item in _predict_iter(df):
                if item[0] == "step":
                    yield json.dumps({"label": item[1], "done": item[2], "total": item[3]}) + "\n"
                else:
                    _, rows, flat = item
                    yield json.dumps({"result": {"rows": rows, "n": len(rows), "run_id": _store_run(flat, blocks),
                                                 "n_valid": int(sum(r["valid"] for r in rows))}}) + "\n"
        except Exception as ex:                       # a half-written stream needs an explicit error line
            yield json.dumps({"error": f"{type(ex).__name__}: {ex}"}) + "\n"

    # no-transform stops any proxy buffering the stream and defeating the progress bar
    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.post("/api/download_sdf")
async def download_sdf(payload: dict):
    """Return the selected rows of one run (payload `run`, the run_id) as an SDF, generated locally
    and never written to the repo. An unknown or expired run id gets HTTP 404.

    When the run came from a dropped SDF, each record is re-emitted **verbatim** — same atoms,
    coordinates, charges, stereo flags and original data fields — and only the prediction fields are
    appended, so a round trip through the app cannot alter a structure. When the run came from a csv
    there is no original record, so a 2D depiction is generated from the SMILES instead.
    """
    run = _get_run(payload.get("run"))
    flat, blocks = run["flat"], run["blocks"]
    rows = payload.get("rows") or []
    if not rows:
        raise HTTPException(400, "no rows selected")
    n = len(flat)
    out, made = [], 0
    for item in rows:
        i = int(item.get("idx", -1))
        if not 0 <= i < n:
            raise HTTPException(400, f"row index {i} is outside this run (0..{n - 1})")
        rec = flat.iloc[i]
        block = blocks[i] if blocks and i < len(blocks) else None
        if block is None:
            # csv input: no original record exists, so depict the molecule from its SMILES
            mol = Chem.MolFromSmiles(str(rec["smiles"]))
            if mol is None:
                continue
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", str(rec["compound"]))
            block = Chem.MolToMolBlock(mol)
            made += 1
        fields = {c: rec[c] for c in flat.columns if c not in ("compound", "smiles")}
        fields.update({k: v for k, v in item.items() if k in ("mpo", "score") and v not in (None, "")})
        out.append(_sdf_record(block, fields))
    body = "".join(out)
    print(f"> SDF export: {len(out)} records ({made} depicted from SMILES, "
          f"{len(out) - made} original records preserved)", flush=True)
    return Response(body, media_type="chemical/x-mdl-sdfile",
                    headers={"Content-Disposition": "attachment; filename=adme_predictions.sdf"})


def _sdf_record(block, fields):
    """One SDF record: the molblock verbatim, then the prediction fields, then the $$$$ terminator."""
    parts = [block if block.endswith("\n") else block + "\n"]
    for k, v in fields.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        parts.append(f"> <{k}>\n{v}\n\n")
    parts.append("$$$$\n")
    return "".join(parts)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    # 127.0.0.1 ONLY — see module docstring (chemistry data must not leave the box).
    uvicorn.run(app, host=WEBAPP.get("host", "127.0.0.1"), port=int(WEBAPP.get("port", 8050)))
