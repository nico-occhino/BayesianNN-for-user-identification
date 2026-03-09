"""
utils.py — Shared infrastructure for the BioVid speaker identification project.

Dataset facts (from README + Metadata.json):
    Videos      : 650 total, 43 users, avg 15.1 per user
    Resolution  : 256×128 px  (width × height) — pre-cropped lip region
    Duration    : avg 1.77s | min 0.53s | max 3.27s
    Correct pwd : 250 videos (word = FLAG)
    Wrong pwd   : 400 videos (word = any other)

Filename convention: VID_<YYYYMMDD>_<HHMMSS>_<WORD>.mp4
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple

# ── Ensure src/ is always importable ─────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FILENAME PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_word_from_filename(filename: str) -> str:
    """
    Extracts the spoken word from a BioVid filename.

    VID_20240425_123033_FLAG.mp4   → "FLAG"
    VID_20240425_123038_pillow.mp4 → "PILLOW"
    """
    stem  = os.path.splitext(filename)[0]   # drop extension
    parts = stem.split("_")
    return parts[-1].upper() if len(parts) >= 4 else "UNKNOWN"


def is_correct_password(word: str) -> bool:
    """FLAG is the correct password in BioVid."""
    return word.upper() == "FLAG"


# ─────────────────────────────────────────────────────────────────────────────
# DATA DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def create_dataframe(
    root_dir:  str,
    filter_by: Optional[str] = None,   # None | "correct" | "wrong"
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Builds a catalog DataFrame from the BioVid dataset directory.

    Args:
        root_dir  : path to dataset root (one subfolder per user)
        filter_by : None       → all 650 videos
                    "correct"  → FLAG only  (250 videos)
                    "wrong"    → non-FLAG   (400 videos)

    Returns:
        df            : DataFrame columns:
                            path       — absolute path to .mp4
                            label      — integer identity 0..N-1
                            user       — username string
                            word       — spoken word (uppercase)
                            is_correct — True if word == FLAG
        label_to_user : dict int → username
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Data directory not found: '{root_dir}'")

    user_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ])

    if not user_folders:
        raise ValueError(f"No user sub-folders found inside '{root_dir}'")

    print(f"📂 Found {len(user_folders)} user folders. Building catalog...")

    data = []
    for label_code, user_folder in enumerate(user_folders):
        user_path   = os.path.join(root_dir, user_folder)
        video_files = sorted([
            f for f in os.listdir(user_path)
            if f.lower().endswith(".mp4")
        ])

        if not video_files:
            print(f"  ⚠️  No .mp4 files in '{user_folder}' — skipping.")
            continue

        for video_file in video_files:
            word = parse_word_from_filename(video_file)
            data.append({
                "path":       os.path.join(user_path, video_file),
                "label":      label_code,
                "user":       user_folder,
                "word":       word,
                "is_correct": is_correct_password(word),
            })

    if not data:
        raise ValueError("No valid .mp4 files found in any user folder.")

    df = pd.DataFrame(data)

    # ── Apply filter ──────────────────────────────────────────────────────────
    if filter_by == "correct":
        df = df[df["is_correct"]].reset_index(drop=True)
    elif filter_by == "wrong":
        df = df[~df["is_correct"]].reset_index(drop=True)

    # ── Re-assign dense label codes after filtering ───────────────────────────
    unique_users  = sorted(df["user"].unique())
    user_to_label = {u: i for i, u in enumerate(unique_users)}
    df["label"]   = df["user"].map(user_to_label)
    label_to_user = {i: u for u, i in user_to_label.items()}

    n_correct = int(df["is_correct"].sum())
    n_wrong   = int((~df["is_correct"]).sum())
    print(f"✓ Catalog ready: {len(df)} videos | "
          f"{df['label'].nunique()} identities | "
          f"avg {len(df)/df['label'].nunique():.1f} videos/user | "
          f"correct={n_correct} wrong={n_wrong}")

    return df, label_to_user


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED K-FOLD
# ─────────────────────────────────────────────────────────────────────────────

def get_folds(
    df:           pd.DataFrame,
    n_splits:     int = 5,
    random_state: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Stratified K-Fold split at video level."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df["path"], df["label"])
    ):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))
        print(f"  Fold {fold_idx+1}: train={len(train_df)} | "
              f"val={len(val_df)} | users in val={val_df['label'].nunique()}")
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leak(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    overlap = set(train_df["path"]) & set(val_df["path"])
    assert len(overlap) == 0, (
        f"🚨 DATA LEAK: {len(overlap)} videos in both splits.\n"
        f"  Examples: {list(overlap)[:3]}"
    )
    print("✓ No data leak confirmed.")


def verify_class_balance(df: pd.DataFrame, split_name: str = "") -> None:
    counts = df.groupby("label").size()
    print(f"  {split_name} class balance → "
          f"min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate for a binary task."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr     = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float(np.mean([fpr[eer_idx], fnr[eer_idx]]))


def compute_multiclass_eer(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    n_classes:  int,
) -> float:
    """Macro-averaged EER (one-vs-rest) across all identity classes."""
    eers = []
    for c in range(n_classes):
        binary = (all_labels == c).astype(int)
        if binary.sum() == 0:
            continue
        eers.append(compute_eer(binary, all_probs[:, c]))
    return float(np.mean(eers))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   List[float],
    val_accs:     List[float],
    fold_idx:     int,
    save_dir:     str = "results",
    mode:         str = "",
) -> None:
    """Saves loss and accuracy curves for one fold."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    suffix = f" [{mode}]" if mode else ""

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title(f"Loss — Fold {fold_idx+1}{suffix}")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title(f"Accuracy — Fold {fold_idx+1}{suffix}")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"curves_fold{fold_idx+1}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  📊 Curves saved → {path}")


def plot_confusion_matrix(
    all_labels:    np.ndarray,
    all_preds:     np.ndarray,
    label_to_user: Dict[int, str],
    save_dir:      str = "results",
    filename:      str = "confusion_matrix.png",
) -> None:
    """Saves a confusion matrix heatmap over all folds."""
    os.makedirs(save_dir, exist_ok=True)
    n      = len(label_to_user)
    cm     = confusion_matrix(all_labels, all_preds, labels=list(range(n)))
    names  = [label_to_user[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (all folds)")
    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {path}")


def plot_modality_comparison(
    results:  Dict[str, Dict],
    save_dir: str = "results",
) -> None:
    """
    Bar chart comparing MC accuracy and EER across modalities.

    Args:
        results : {
            "audio":      {"mc_acc": 0.37, "eer": 0.07,
                           "mc_acc_std": 0.02, "eer_std": 0.01},
            "video":      {...},
            "multimodal": {...},
        }
    """
    os.makedirs(save_dir, exist_ok=True)
    modes   = list(results.keys())
    accs    = [results[m]["mc_acc"]              for m in modes]
    eers    = [results[m]["eer"]                 for m in modes]
    acc_std = [results[m].get("mc_acc_std", 0)   for m in modes]
    eer_std = [results[m].get("eer_std",    0)   for m in modes]
    colors  = ["#4C72B0", "#DD8452", "#55A868",  "#C44E52"][:len(modes)]

    x = np.arange(len(modes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.bar(x, accs, yerr=acc_std, capsize=5, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in modes])
    ax1.set_ylabel("MC Accuracy")
    ax1.set_title("MC Accuracy by Modality")
    ax1.set_ylim(0, 1)
    for i, (v, s) in enumerate(zip(accs, acc_std)):
        ax1.text(i, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(x, eers, yerr=eer_std, capsize=5, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in modes])
    ax2.set_ylabel("EER")
    ax2.set_title("EER by Modality (lower is better)")
    for i, (v, s) in enumerate(zip(eers, eer_std)):
        ax2.text(i, v + s + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "modality_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Modality comparison saved → {path}")