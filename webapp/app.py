#!/usr/bin/env python3
"""app.py — Local-only FastAPI server for drag-drop ADME property prediction.

LOCALHOST ONLY. This app handles SMILES / compound ids / predicted values, so it
MUST NOT be exposed off this machine — it binds to 127.0.0.1 (see config webapp.host).

Pipeline (reusing the MLTrail vault + the ADME config):
  startup -> discover the deployed RF models in the MLTrail vault (prefer new H237 over H236)
          -> load each estimator + trained feature columns + conf_recal calibration (from the bundle)
          -> fallback scale = std of the archived training-set labels when no calibration is bundled
          -> load any config `webapp.extra_models` (non-ADME MLTrail models, own grid column)
  upload  -> read the dropped SDF/CSV once -> featurize H237 once
          -> per model: mean (raw value) + per-tree std (-> conf_recal confidence) -> favorable flag
          -> per extra classifier: P(positive class) + the decision margin |2p-1| as the confidence
  render  -> LiveDesign-style grid, one diagonal-split cell per endpoint

Predictions/SMILES/ids render only in the LOCAL browser (localhost); nothing crosses to any cloud.
"""
import atexit
import io
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from mltrail import Registry
from mltrail.backends import align_features
from mltrail.featurizers import get_featurizer
from mltrail.readers import read_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
WEBAPP = CONFIG.get("webapp", {})
ENDPOINT_CFG = CONFIG["ADME_ENDPOINTS"]                 # transform + unit per endpoint

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

CHAMPIONS = {}          # endpoint -> {model, feature_cols, sigma, transform, unit, cutoff, favorable, model_id}
EXTRAS = {}             # key -> {model, feature_cols, pos_idx, label, unit, threshold, model_id} (extra columns)
_FEATURIZE = None       # H236 featurizer (built once at startup)
_LAST_CSV = None        # flat DataFrame of the most recent prediction run (for /api/download)


# ---------- startup: discover + load the champions ----------

def _discover_champions():
    """Load the deployed RF models from the MLTrail vault, per endpoint. Prefer the new H237 models
    (`adme_<ep>_h237`, conf_recal calibration bundled) over the older H236 champions (`adme_<ep>`,
    training-label-std fallback). Loads model + feature_cols + calibration from each model bundle."""
    reg = Registry.from_default()
    cutoffs = WEBAPP.get("cutoffs", {})
    # gather adme_ sklearn models keyed by (endpoint, features_type); prefer H237 below
    found = {}
    for _, row in reg.list().iterrows():
        d = reg.details(row["id"])
        name = str(d.get("experiment_name", ""))
        if not (name.startswith("adme_") and d.get("framework") == "sklearn"):
            continue
        ft = d.get("features_type")
        if ft not in ("H236", "H237"):
            continue
        ep = name[len("adme_"):].removesuffix("_h237")
        found.setdefault(ep, {})[ft] = d
    for ep, by_ft in found.items():
        d = by_ft.get("H237") or by_ft.get("H236")
        ft = "H237" if "H237" in by_ft else "H236"
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


def _load_extras():
    """Load the config `webapp.extra_models` — MLTrail models that are NOT ADME endpoints (e.g. the
    Px single/low activity classifier, id 19). They get their own grid column and their own CSV columns,
    but they stay OUT of the MPO formula scope. Classification only: the column shows P(positive class)."""
    reg = Registry.from_default()
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


def _read_upload(path, filename):
    """Read one dropped file into df[compound, smiles], auto-detecting the columns (csv) or
    deriving SMILES from structure (sdf). SDF uses the molecule title (_Name) as compound id."""
    ext = Path(filename).suffix.lower()
    if ext == ".sdf":
        return read_dataset(path, compound_id="_Name")
    head = pd.read_csv(path, nrows=0)
    cols = list(head.columns)
    smi_col = next((c for c in cols if "smiles" in c.lower()), None)
    if smi_col is None:
        raise HTTPException(400, f"{filename}: no SMILES column found (looked for a name containing 'smiles')")
    id_col = next((c for c in cols if any(k in c.lower() for k in ("compound", "_id", "id", "name"))), None)
    return read_dataset(path, smiles_column=smi_col, compound_id=id_col)


def _predict_all(df):
    """Featurize H236 once, score every champion, and return (rows, flat_df).

    rows: per-compound dicts for the grid (compound, smiles, svg, valid, preds{ep:{value,favorable,confidence}}).
    flat_df: the same data flattened to one row per compound for CSV download.
    """
    df = df.reset_index(drop=True)
    df["_key"] = np.arange(len(df))
    feats = _FEATURIZE(pd.DataFrame({"compound": df["_key"].values, "smiles": df["smiles"].values}))
    keys = feats["compound"].to_numpy()                     # _key of each row that featurized (bad SMILES dropped)
    pos = {k: i for i, k in enumerate(keys)}

    # per endpoint: mean (raw) + confidence for every valid row, indexed by _key
    ep_out = {}
    for ep, c in CHAMPIONS.items():
        X = align_features(feats, c["feature_cols"])
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)   # drop names: trees were fitted on a plain array
        per_tree = np.stack([est.predict(Xv) for est in c["model"].estimators_])   # (n_trees, n_valid)
        raw = _INV[c["transform"]](per_tree.mean(axis=0))
        conf = _confidence(per_tree.std(axis=0), c)                                # (0, 1], conf_recal or fallback
        ep_out[ep] = (raw, conf)

    # extra classifiers: value = P(positive class); confidence = decision margin |2p-1| (0 at the
    # 0.5 boundary, 1 at a unanimous forest). NOTE: a per-tree std is useless here — the trees vote 0/1,
    # so std ~ sqrt(p(1-p)) and carries no information beyond p itself.
    for xk, c in EXTRAS.items():
        X = align_features(feats, c["feature_cols"])
        Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        prob = c["model"].predict_proba(Xv)[:, c["pos_idx"]]
        ep_out[xk] = (prob, np.abs(2.0 * prob - 1.0))

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
    return rows, pd.DataFrame(flat)


# ---------- app ----------

@asynccontextmanager
async def lifespan(app):
    global _FEATURIZE
    _discover_champions()                 # load the deployed models + per-endpoint calibration
    _load_extras()                        # non-ADME extra-column models (config webapp.extra_models)
    _FEATURIZE = get_featurizer("H237", CONFIG)   # H237 featurizer (H237 columns cover H236 fallbacks)
    yield


app = FastAPI(title="ADME property prediction", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
def models():
    """Champion roster (endpoint, unit, cutoff, favorable direction) + the cell palette. Metadata only."""
    order = _ordered_endpoints()
    eps = [{"key": ep, "unit": CHAMPIONS[ep]["unit"], "cutoff": CHAMPIONS[ep]["cutoff"],
            "favorable": CHAMPIONS[ep]["favorable"], "transform": CHAMPIONS[ep]["transform"],
            "color_slope": COLOR_SLOPE, "model_id": CHAMPIONS[ep]["model_id"]} for ep in order]
    # a probability lives on a 0-1 scale, so its fade needs a steeper slope than a log-unit endpoint
    extras = [{"key": k, "label": c["label"], "unit": c["unit"], "cutoff": c["threshold"],
               "favorable": ">=", "kind": "classification", "transform": "identity",
               "color_slope": c["color_slope"], "model_id": c["model_id"]}
              for k, c in EXTRAS.items()]
    return {"endpoints": eps, "extras": extras, "palette": PALETTE}


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    """Accept one or more dropped .sdf/.csv files; return per-compound predictions for the grid."""
    global _LAST_CSV
    frames = []
    for uf in files:
        staged = STAGE_DIR / f"{uuid.uuid4().hex}{Path(uf.filename).suffix.lower()}"
        staged.write_bytes(await uf.read())
        try:
            frames.append(_read_upload(str(staged), uf.filename))
        finally:
            staged.unlink(missing_ok=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["compound", "smiles"])
    if len(df) == 0:
        return {"rows": [], "n": 0, "n_valid": 0}
    rows, _LAST_CSV = _predict_all(df)
    return {"rows": rows, "n": len(rows), "n_valid": int(sum(r["valid"] for r in rows))}


@app.get("/api/download")
def download():
    """Return the most recent prediction run as a CSV (generated locally, never written to the repo)."""
    if _LAST_CSV is None:
        raise HTTPException(404, "no predictions yet")
    buf = io.StringIO(); _LAST_CSV.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=adme_predictions.csv"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    # 127.0.0.1 ONLY — see module docstring (chemistry data must not leave the box).
    uvicorn.run(app, host=WEBAPP.get("host", "127.0.0.1"), port=int(WEBAPP.get("port", 8050)))
