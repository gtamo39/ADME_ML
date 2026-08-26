#!/usr/bin/env python3
"""app.py — Local-only FastAPI server for drag-drop ADME property prediction.

LOCALHOST ONLY. This app handles SMILES / compound ids / predicted values, so it
MUST NOT be exposed off this machine — it binds to 127.0.0.1 (see config webapp.host).

Pipeline (reusing the MLTrail vault + the ADME config):
  startup -> discover the 8 champion RF (H236) models in the MLTrail vault
          -> load each estimator + its trained feature columns
          -> per endpoint, sigma = std of the archived training-set labels (modelling space)
  upload  -> read the dropped SDF/CSV once -> featurize H236 once
          -> per model: mean (raw value) + per-tree std (-> confidence) -> favorable flag
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

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from mltrail import Registry
from mltrail.backends import align_features, load_model
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
_FEATURIZE = None       # H236 featurizer (built once at startup)
_LAST_CSV = None        # flat DataFrame of the most recent prediction run (for /api/download)


# ---------- startup: discover + load the champions ----------

def _discover_champions():
    """Load the 8 champion RF (H236) models from the MLTrail vault and their per-endpoint scale."""
    reg = Registry.from_default()
    cutoffs = WEBAPP.get("cutoffs", {})
    listing = reg.list()
    for _, row in listing.iterrows():
        d = reg.details(row["id"])
        name = str(d.get("experiment_name", ""))
        if not (name.startswith("adme_") and d.get("features_type") == "H236"
                and d.get("framework") == "sklearn"):
            continue
        ep = name[len("adme_"):]
        model, feature_cols = load_model(d["framework"], d["model_path"])
        # sigma (v1) = std of the archived training-set labels (modelling space) — see wiki reminder
        try:
            sigma = float(reg.load_training_set(d["id"])["label"].std())
        except Exception as ex:
            print(f"WARN: {ep}: training-label std unavailable ({ex}); sigma=1.0", flush=True)
            sigma = 1.0
        cut = cutoffs.get(ep, {})
        CHAMPIONS[ep] = {
            "model": model, "feature_cols": feature_cols, "sigma": sigma or 1.0,
            "transform": ENDPOINT_CFG[ep]["transform"], "unit": ENDPOINT_CFG[ep].get("unit", ""),
            "cutoff": cut.get("value"), "favorable": cut.get("favorable"), "model_id": int(d["id"]),
        }
    print(f"> loaded {len(CHAMPIONS)} champion RF (H236) models: {sorted(CHAMPIONS)}", flush=True)


def _ordered_endpoints():
    """Endpoints in the configured column order, restricted to the champions actually loaded."""
    order = WEBAPP.get("endpoint_order", sorted(CHAMPIONS))
    return [e for e in order if e in CHAMPIONS] + [e for e in CHAMPIONS if e not in order]


# ---------- helpers ----------

def _svg(smiles, width=150, height=100):
    """Render a compound to an inline SVG string (local, RDKit). Empty string on parse failure."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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
        conf = np.exp(-per_tree.std(axis=0) / c["sigma"])                          # (0, 1], mirrors uq_std_to_confidence
        ep_out[ep] = (raw, conf)

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
        rows.append({"compound": r["compound"], "smiles": smi, "svg": _svg(smi), "valid": valid, "preds": preds})
        flat.append(flat_row)
    return rows, pd.DataFrame(flat)


# ---------- app ----------

@asynccontextmanager
async def lifespan(app):
    global _FEATURIZE
    _discover_champions()                 # load the champion models + per-endpoint sigma
    _FEATURIZE = get_featurizer("H236", CONFIG)   # build the H236 featurizer once
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
            "model_id": CHAMPIONS[ep]["model_id"]} for ep in order]
    return {"endpoints": eps, "palette": PALETTE}


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
