"""Harmonize public + predicted solubility datasets into one tidy table.

Output columns:
    smiles      : str   — as provided by each source (not re-canonicalized)
    origin      : str   — source dataset, e.g. 'ChEMBL', 'AqSolDB', 'PharmaBench'
    type        : str   — measurement class (see TYPES below)
    solubility  : float — solubility in µM, matched to the in-house target
                          'Thermodynamic Solubility (1) (µM)' in 20260625_thermoSol.csv

Measurement types:
    thermodynamic_experimental, kinetic_experimental, mixed (experimental, assay
    type not resolvable), thermodynamic_predicted.

Unit conversions to µM (S = solubility in mol/L):
    logS = log10(mol/L)   ->  10**logS * 1e6
    log10(nM)             ->  10**v / 1e3
    nM                    ->  v / 1e3
    µg/mL                 ->  v * 1e3 / MW        (MW from RDKit)
    log10(µg/mL)          ->  10**v * 1e3 / MW

Censored values (ChEMBL '<'/'>') are kept at their reported bound, matching how
the in-house target parses '< 2' -> 2.0. Pass keep_relation=True to retain the flag.

Usage in a notebook:
    from harmonize_sol import harmonize_solubility
    df = harmonize_solubility()                 # all default sources -> long table
    df = harmonize_solubility(out_csv='data/public_solubility/harmonized_sol.csv')
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

DATA_DIR = "data"
SOL_DIR = "public_solubility"


# ── unit converters (all return µM) ───────────────────────────────────────────
def _from_logS(v):       return np.power(10.0, v) * 1e6
def _from_log10nM(v):    return np.power(10.0, v) / 1e3
def _from_nM(v):         return v / 1e3
def _from_ugml(v, mw):   return v * 1e3 / mw
def _from_log10ugml(v, mw): return np.power(10.0, v) * 1e3 / mw


@lru_cache(maxsize=None)
def _molwt(smiles):
    """RDKit molecular weight (g/mol); NaN if SMILES is unparseable."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Descriptors.MolWt(m) if m else np.nan


def _frame(smiles, solubility, origin, mtype):
    """Assemble a standard 4-column frame and drop rows with no SMILES."""
    out = pd.DataFrame({"smiles": smiles, "solubility": solubility})
    out["origin"], out["type"] = origin, mtype
    return out[out["smiles"].notna() & (out["smiles"].astype(str).str.len() > 0)]


# ── per-source loaders ─────────────────────────────────────────────────────────
def _load_solcuration(path, origin, mtype):
    """SolCuration clean/cure files: columns smiles, logS, weight (logS = mol/L)."""
    df = pd.read_csv(path)
    return _frame(df["smiles"], _from_logS(df["logS"]), origin, mtype)


def _classify_chembl(desc):
    """Resolve ChEMBL measurement type from the assay description text."""
    d = str(desc).lower()
    if "kinetic" in d:        return "kinetic_experimental"
    if "thermodynamic" in d:  return "thermodynamic_experimental"
    return "mixed"


def _load_chembl(path, origin="ChEMBL"):
    """Fresh ChEMBL API pull; convert nM and µg/mL to µM, type per assay description."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df["canonical_smiles"].notna()].copy()
    val = pd.to_numeric(df["standard_value"], errors="coerce")
    units = df["standard_units"].astype(str)
    sol = pd.Series(np.nan, index=df.index)
    sol[units == "nM"] = _from_nM(val[units == "nM"])
    ug = units == "ug.mL-1"
    mw = df.loc[ug, "canonical_smiles"].map(_molwt)
    sol[ug] = _from_ugml(val[ug], mw)
    out = _frame(df["canonical_smiles"], sol, origin, df["assay_description"].map(_classify_chembl))
    return out[out["solubility"].notna()]


def _load_pharmabench(path, origin="PharmaBench"):
    """PharmaBench final water-solubility set; value = log10(nM), equilibrium-filtered."""
    df = pd.read_csv(path)
    return _frame(df["Smiles_unify"], _from_log10nM(df["value"]), origin, "thermodynamic_experimental")


def _load_biogen(path, origin="Biogen-Fang"):
    """Biogen Fang 2023 ADME; solubility = log10(µg/mL) at pH 6.8."""
    df = pd.read_csv(path)
    col = "LOG SOLUBILITY PH 6.8 (ug/mL)"
    v = pd.to_numeric(df[col], errors="coerce")
    mw = df["SMILES"].map(_molwt)
    return _frame(df["SMILES"], _from_log10ugml(v, mw), origin, "thermodynamic_experimental")


def _load_solchallenge(path):
    """Solubility Challenge 2 gold standard; intrinsic S0, µM precomputed.
    Keeps per-row origin (SolChallenge-SET1 train / SolChallenge-SET2 test)."""
    df = pd.read_csv(path)
    df = df[df["smiles"].astype(str).str.len() > 0]
    out = df[["smiles", "origin", "type", "solubility"]].copy()
    return out[out["solubility"] > 0]


def _load_wiki_ps0(path, origin="Wiki-pS0"):
    """Wiki-pS0 bRo5 set (Avdeef 2020); intrinsic S0, already converted to µM."""
    df = pd.read_csv(path)
    df = df[df["smiles"].astype(str).str.len() > 0]
    return _frame(df["smiles"], df["solubility"], origin, "thermodynamic_experimental")


def _load_patentdb(path, origin="PROTAC-PatentDB"):
    """PROTAC-PatentDB (ADMETlab 3.0 predicted); logS = log10(mol/L)."""
    df = pd.read_excel(path, usecols=["SMILES", "logS"])
    return _frame(df["SMILES"], _from_logS(pd.to_numeric(df["logS"], errors="coerce")),
                  origin, "thermodynamic_predicted")


# ── source registry ────────────────────────────────────────────────────────────
def _default_sources(data_dir):
    sc = os.path.join(data_dir, SOL_DIR, "aqsoldbc_solcuration", "cure")
    p = lambda *a: os.path.join(data_dir, SOL_DIR, *a)
    return [
        ("ChEMBL",          _load_chembl,      p("chembl_api", "chembl_solubility_raw.csv")),
        ("AqSolDB",         _load_solcuration, (os.path.join(sc, "aqsol_cure.csv"), "AqSolDB", "mixed")),
        ("AQUA",            _load_solcuration, (os.path.join(sc, "aqua_cure.csv"), "AQUA", "thermodynamic_experimental")),
        ("PHYS",            _load_solcuration, (os.path.join(sc, "phys_cure.csv"), "PHYS", "thermodynamic_experimental")),
        ("ESOL",            _load_solcuration, (os.path.join(sc, "esol_cure.csv"), "ESOL", "thermodynamic_experimental")),
        ("OChem",           _load_solcuration, (os.path.join(sc, "ochem_cure.csv"), "OChem", "mixed")),
        ("KINECT",          _load_solcuration, (os.path.join(sc, "kinect_cure.csv"), "KINECT", "kinetic_experimental")),
        ("PharmaBench",     _load_pharmabench, p("pharmabench", "water_sol_reg_final_data.csv")),
        ("Biogen-Fang",     _load_biogen,      p("biogen_fang_adme", "ADME_public_set_3521.csv")),
        ("Wiki-pS0",        _load_wiki_ps0,    p("wiki_ps0_bigmol", "wiki_ps0_31_bigmol.csv")),
        ("SolChallenge",    _load_solchallenge, p("solubility_challenge", "solchallenge_gold.csv")),
        ("PROTAC-PatentDB", _load_patentdb,    p("protac_patent", "PROTAC_Patent_Compounds.xlsx")),
    ]


def harmonize_solubility(data_dir=DATA_DIR, include=None, exclude=None,
                         collapse_per_type=False, add_inchikey=False, out_csv=None):
    """Load every source, convert to µM, and return one tidy long table.

    Args:
        data_dir: repo data folder (default 'data'; notebook %cd's to repo root).
        include / exclude: optional source-name lists to subset the registry.
        collapse_per_type: group to one row per (InChIKey, origin, type) using the
            median µM — deduplicates repeat measurements within a source.
        add_inchikey: add an 'inchikey' column (also implied by collapse_per_type).
        out_csv: if set, write the result to this path.

    Returns:
        DataFrame[smiles, origin, type, solubility(µM)] (+ inchikey if requested).
    """
    frames = []
    for name, loader, spec in _default_sources(data_dir):
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        args = spec if isinstance(spec, tuple) else (spec,)
        frames.append(loader(*args))

    df = pd.concat(frames, ignore_index=True)[["smiles", "origin", "type", "solubility"]]
    # solubility must be positive to be physical; drop the rest
    df = df[df["solubility"] > 0]

    if add_inchikey or collapse_per_type:
        df["inchikey"] = df["smiles"].map(_inchikey)
    if collapse_per_type:
        df = (df[df["solubility"].notna()]
              .groupby(["inchikey", "origin", "type"], as_index=False)
              .agg(smiles=("smiles", "first"), solubility=("solubility", "median")))
        df = df[["smiles", "origin", "type", "solubility", "inchikey"]]

    if out_csv:
        df.to_csv(out_csv, index=False)
    return df.reset_index(drop=True)


@lru_cache(maxsize=None)
def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(m) if m else np.nan
