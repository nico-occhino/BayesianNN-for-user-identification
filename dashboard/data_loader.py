"""
data_loader.py — Shared data loading utilities for BioVid dashboards.

Expected folder layout (results/ at PROJECT ROOT, one level above dashboard/):
    project_root/
    ├── dashboard/
    │   ├── data_loader.py
    │   ├── overview_dashboard.py
    │   └── pages/
    └── results/
        ├── audio/         metrics.json  history_fold*.csv  predictions_fold*.csv  *.png
        ├── audio_paper/
        ├── video/
        └── multimodal/
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Resolve BASE_DIR: project root = dashboard/../  ────────────────────────
_THIS       = Path(__file__).resolve()
BASE_DIR    = _THIS.parent.parent          # ← THE FIX (was .parent)
RESULTS_DIR = BASE_DIR / "results"

# ── Canonical experiment metadata ──────────────────────────────────────────
EXPERIMENTS = {
    "Audio Lightweight": {
        "folder":   "audio",
        "modality": "Audio",
        "params":   "153K",
        "color":    "#EF553B",
    },
    "Audio Paper-Exact": {
        "folder":   "audio_paper",
        "modality": "Audio",
        "params":   "25M",
        "color":    "#FF7F0E",
    },
    "Video Lightweight": {
        "folder":   "video",
        "modality": "Video",
        "params":   "153K",
        "color":    "#00CC96",
    },
    "Multimodal Lightweight": {
        "folder":   "multimodal",
        "modality": "Multimodal",
        "params":   "307K",
        "color":    "#636EFA",
    },
}

MODEL_NAMES  = list(EXPERIMENTS.keys())
MODEL_COLORS = {k: v["color"] for k, v in EXPERIMENTS.items()}


def _exp_dir(model_name: str) -> Path:
    return RESULTS_DIR / EXPERIMENTS[model_name]["folder"]


# ── Availability check ─────────────────────────────────────────────────────
def check_availability() -> dict:
    return {
        name: (_exp_dir(name) / "metrics.json").exists()
        for name in MODEL_NAMES
    }


# ── metrics.json loaders ───────────────────────────────────────────────────
@st.cache_data
def load_metrics(model_name: str):
    path = _exp_dir(model_name) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_all_metrics() -> dict:
    return {name: load_metrics(name) for name in MODEL_NAMES}


@st.cache_data
def build_summary_df() -> pd.DataFrame:
    """
    Per-model summary DataFrame.
    Columns: Model, Modality, Parameters, MC_Accuracy, Std, EER, Std_EER
    """
    rows = []
    for name, meta in EXPERIMENTS.items():
        m = load_metrics(name)
        if m is None:
            continue
        s = m["summary"]
        rows.append({
            "Model":       name,
            "Modality":    meta["modality"],
            "Parameters":  meta["params"],
            "MC_Accuracy": round(s["mean_mc_acc"], 4),
            "Std":         round(s["std_mc_acc"],  4),
            "EER":         round(s["mean_eer"],    4),
            "Std_EER":     round(s["std_eer"],     4),
        })
    return pd.DataFrame(rows)


@st.cache_data
def build_fold_df() -> pd.DataFrame:
    """
    Per-fold DataFrame.
    Columns: Model, Modality, Fold, Fold_num, MC_Accuracy, EER
    """
    rows = []
    for name, meta in EXPERIMENTS.items():
        m = load_metrics(name)
        if m is None:
            continue
        for fold in m["folds"]:
            rows.append({
                "Model":       name,
                "Modality":    meta["modality"],
                "Fold":        f"Fold {fold['fold']}",
                "Fold_num":    fold["fold"],
                "MC_Accuracy": round(fold["mc_acc"], 4),
                "EER":         round(fold["eer"],    6),
            })
    return pd.DataFrame(rows)


# ── History CSV loaders ────────────────────────────────────────────────────
def _normalise_history_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    renames = {}
    for c in ["train_loss", "loss"]:
        if c in df.columns and "train_loss" not in renames.values():
            renames[c] = "train_loss"
    for c in ["val_loss", "validation_loss"]:
        if c in df.columns and "val_loss" not in renames.values():
            renames[c] = "val_loss"
    for c in ["train_acc", "train_accuracy", "accuracy", "acc"]:
        if c in df.columns and "train_acc" not in renames.values():
            renames[c] = "train_acc"
    for c in ["val_acc", "val_accuracy", "validation_accuracy"]:
        if c in df.columns and "val_acc" not in renames.values():
            renames[c] = "val_acc"
    return df.rename(columns=renames)


@st.cache_data
def load_history(model_name: str, fold: int):
    path = _exp_dir(model_name) / f"history_fold{fold}.csv"
    if not path.exists():
        return None
    try:
        return _normalise_history_cols(pd.read_csv(path))
    except Exception:
        return None


@st.cache_data
def load_all_histories(model_name: str) -> dict:
    return {fold: load_history(model_name, fold) for fold in range(1, 6)}


# ── Predictions CSV loader ─────────────────────────────────────────────────
@st.cache_data
def load_predictions(model_name: str, fold: int):
    path = _exp_dir(model_name) / f"predictions_fold{fold}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── Image path helpers ─────────────────────────────────────────────────────
def get_confusion_matrix_path(model_name: str):
    p = _exp_dir(model_name) / "confusion_matrix_all_folds.png"
    return p if p.exists() else None


def get_curves_path(model_name: str, fold: int):
    p = _exp_dir(model_name) / f"curves_fold{fold}.png"
    return p if p.exists() else None


# ── Streamlit helpers ──────────────────────────────────────────────────────
def missing_data_warning(model_name: str) -> None:
    folder = EXPERIMENTS[model_name]["folder"]
    st.warning(
        f"⚠️ **{model_name}**: `results/{folder}/metrics.json` not found. "
        f"Copy Kaggle output files into `results/{folder}/` to populate this view.",
        icon="📂",
    )


def sidebar_results_status() -> None:
    avail = check_availability()
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Results status")
    for name, ok in avail.items():
        icon = "✅" if ok else "❌"
        short = name.replace(" Lightweight", " LW").replace("Paper-Exact", "Paper")
        st.sidebar.markdown(f"{icon} {short}")