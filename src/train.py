"""
train.py — BioVid Speaker Identification Training Script

Supports three experiments via --mode flag:

    python train.py --mode audio       → Experiment 1: Mel-Spectrogram CNN
    python train.py --mode video       → Experiment 2: Lip frame CNN
    python train.py --mode multimodal  → Experiment 3: Late fusion (audio+video)

All experiments use:
    - Stratified 5-fold cross-validation (video-level, no data leak)
    - MC Dropout inference (T=30 passes) after each fold
    - AMP (Automatic Mixed Precision) for faster GPU training
    - ReduceLROnPlateau scheduler with generous patience
    - Best checkpoint saved per fold
    - Training curves and confusion matrix saved to results_dir

Dataset filters (--filter flag):
    all      → all 650 videos (default)
    correct  → FLAG password only  (250 videos)
    wrong    → wrong password only (400 videos)

Usage examples:
    python train.py --mode audio      --epochs 200 --num_workers 0
    python train.py --mode video      --epochs 200 --num_workers 0
    python train.py --mode multimodal --epochs 200 --num_workers 0
    python train.py --mode audio      --epochs 200 --filter correct
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast

# ── Ensure src/ is always importable ─────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import (
    create_dataframe, get_folds, verify_no_leak,
    compute_multiclass_eer, plot_training_curves,
    plot_confusion_matrix, plot_modality_comparison,
)
from model import AudioModel, VideoModel, FusionModel, PaperAudioModel, mc_predict
from audio_dataset import AudioDataset, get_train_transform
from video_dataset import VideoDataset, MultimodalDataset


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioVid Speaker Identification — Train & Evaluate"
    )

    # Core
    p.add_argument("--mode", choices=["audio", "video", "multimodal"],
                   default="audio")
    p.add_argument("--filter", choices=["all", "correct", "wrong"],
                   default="all",
                   help="Filter videos by password type")

    # Paths
    p.add_argument("--data_dir",    default=os.path.join("..", "data"))
    p.add_argument("--results_dir", default=os.path.join("..", "results"))

    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--n_folds",     type=int,   default=5)
    p.add_argument("--mc_passes",   type=int,   default=30)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--no_cache",    action="store_true")

    # Model
    p.add_argument("--n_classes",   type=int,   default=43)
    p.add_argument("--feature_dim", type=int,   default=128)
    p.add_argument("--dropout_p",   type=float, default=0.4)
    p.add_argument("--n_frames",    type=int,   default=16)
    p.add_argument("--paper",       action="store_true",
                   help="Use paper-exact architecture (Spata et al. 2025): "
                        "4 blocks + Flatten + Dense(512), ~25M params, "
                        "dropout=0.2. Only valid with --mode audio.")

    # LR scheduler
    p.add_argument("--patience",    type=int,   default=25,
                   help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr_factor",   type=float, default=0.5)
    p.add_argument("--min_lr",      type=float, default=1e-5)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(args, train_df, val_df):
    """Returns (train_dataset, val_dataset) for the chosen mode."""
    use_cache = not args.no_cache
    filter_by = None if args.filter == "all" else args.filter

    if args.mode == "audio":
        fixed = (128, 128) if args.paper else None
        train_ds = AudioDataset(
            train_df,
            transform  = get_train_transform(),
            use_cache  = use_cache,
            fixed_size = fixed,
        )
        val_ds = AudioDataset(
            val_df,
            transform  = None,
            use_cache  = use_cache,
            fixed_size = fixed,
        )

    elif args.mode == "video":
        train_ds = VideoDataset(
            train_df,
            n_frames   = args.n_frames,
            use_cache  = use_cache,
        )
        val_ds = VideoDataset(
            val_df,
            n_frames   = args.n_frames,
            use_cache  = use_cache,
        )

    else:  # multimodal
        train_ds = MultimodalDataset(
            train_df,
            n_frames        = args.n_frames,
            transform_audio = get_train_transform(),  # SpecAugment train only
            use_cache       = use_cache,
        )
        val_ds = MultimodalDataset(
            val_df,
            n_frames        = args.n_frames,
            transform_audio = None,                   # clean val
            use_cache       = use_cache,
        )

    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args) -> nn.Module:
    """Returns the model for the chosen mode."""
    if args.paper:
        if args.mode != "audio":
            raise ValueError("--paper is only supported with --mode audio")
        print("  ⚡ Using PAPER-EXACT architecture (Spata et al. 2025)")
        print("     4 Conv blocks + Flatten + Dense(512) | ~25M params | dropout=0.2")
        return PaperAudioModel(n_classes=args.n_classes, dropout_p=0.2)

    kw = dict(
        feature_dim = args.feature_dim,
        n_classes   = args.n_classes,
        p           = args.dropout_p,
    )
    if args.mode == "audio":
        return AudioModel(**kw)
    elif args.mode == "video":
        return VideoModel(**kw)
    else:
        return FusionModel(**kw, hidden=512)


# ─────────────────────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler,
    device:    str,
    mode:      str,
) -> tuple:
    """Runs one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = total_correct = total_n = 0

    for batch in loader:
        optimizer.zero_grad()

        if mode == "multimodal":
            audio, video, labels = batch
            audio  = audio.to(device)
            video  = video.to(device)
            labels = labels.to(device)
            with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                logits = model(audio, video)
                loss   = criterion(logits, labels)
        else:
            inputs, labels = batch
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                logits = model(inputs)
                loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds          = logits.argmax(dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
    mode:      str,
) -> tuple:
    """Evaluates on validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = total_correct = total_n = 0

    for batch in loader:
        if mode == "multimodal":
            audio, video, labels = batch
            audio  = audio.to(device)
            video  = video.to(device)
            labels = labels.to(device)
            with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                logits = model(audio, video)
                loss   = criterion(logits, labels)
        else:
            inputs, labels = batch
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                logits = model(inputs)
                loss   = criterion(logits, labels)

        preds          = logits.argmax(dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


# ─────────────────────────────────────────────────────────────────────────────
# MC INFERENCE ON FULL VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_mc_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
    mode:   str,
    T:      int = 30,
):
    """
    Runs MC inference over the entire validation loader.
    Returns (all_labels, all_preds, all_probs, mean_uncertainty).
    """
    model.eval()
    all_labels = []
    all_probs  = []
    all_unc    = []

    for batch in loader:
        if mode == "multimodal":
            audio, video, labels = batch
            audio  = audio.to(device)
            video  = video.to(device)
            labels = labels.to(device)
            inputs = (audio, video)
        else:
            inputs_raw, labels = batch
            inputs_raw = inputs_raw.to(device)
            labels     = labels.to(device)
            inputs     = (inputs_raw,)

        probs, unc = mc_predict(model, inputs, T=T)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_unc.append(unc.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_unc    = np.concatenate(all_unc)
    all_preds  = all_probs.argmax(axis=1)

    return all_labels, all_preds, all_probs, float(all_unc.mean())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PER-SAMPLE UNCERTAINTY COLLECTION  (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_per_sample_uncertainty(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
    mode:   str,
    T:      int = 30,
) -> np.ndarray:
    """
    Returns per-sample uncertainty as a (N,) float array.
    Uncertainty = mean std-dev across classes over T MC passes.
    This is a second pass over the val set — cheap since cache is warm.
    """
    model.eval()
    all_unc = []

    for batch in loader:
        if mode == "multimodal":
            audio, video, _ = batch
            audio  = audio.to(device)
            video  = video.to(device)
            inputs = (audio, video)
        else:
            inputs_raw, _ = batch
            inputs_raw    = inputs_raw.to(device)
            inputs        = (inputs_raw,)

        # T forward passes
        passes = []
        for _ in range(T):
            import torch.nn.functional as F
            logits = model(*inputs)
            passes.append(F.softmax(logits, dim=-1).unsqueeze(0))

        passes = torch.cat(passes, dim=0)        # (T, B, C)
        unc    = passes.std(dim=0).mean(dim=-1)  # (B,)
        all_unc.append(unc.cpu().numpy())

    return np.concatenate(all_unc)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Header ────────────────────────────────────────────────────────────────
    filter_label = args.filter.upper()
    arch_label   = "PAPER (Spata et al. 2025, ~25M params)" if args.paper else "LIGHTWEIGHT (GAP, ~153K params)"
    print("=" * 60)
    print(f"  BioVid Speaker Identification — {args.mode.upper()} experiment")
    print(f"  Architecture: {arch_label}")
    if args.filter != "all":
        print(f"  Filter      : {filter_label} password videos only")
    print(f"  Device      : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Folds       : {args.n_folds}")
    print(f"  MC passes   : {args.mc_passes}")
    if args.mode in ("video", "multimodal"):
        print(f"  N frames    : {args.n_frames}")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    filter_by = None if args.filter == "all" else args.filter
    print("\nBuilding dataset catalog...")
    df, label_to_user = create_dataframe(args.data_dir, filter_by=filter_by)
    n_classes = df["label"].nunique()

    if args.n_classes != n_classes:
        print(f"  ⚠️  --n_classes={args.n_classes} overridden by actual "
              f"n_classes={n_classes}")
    args.n_classes = n_classes

    print("\nBuilding stratified folds...")
    folds = get_folds(df, n_splits=args.n_folds)

    print("\nVerifying data integrity across all folds...")
    for train_df, val_df in folds:
        verify_no_leak(train_df, val_df)
    print("✓ All folds clean.")

    # ── Results storage ───────────────────────────────────────────────────────
    # Build a unique results subdirectory per experiment variant
    mode_tag = args.mode
    if args.paper:
        mode_tag = f"{args.mode}_paper"
    if args.filter != "all":
        mode_tag = f"{mode_tag}_{args.filter}"
    mode_dir = os.path.join(args.results_dir, mode_tag)
    os.makedirs(mode_dir, exist_ok=True)

    fold_mc_accs = []
    fold_eers    = []
    all_labels_global = []
    all_preds_global  = []

    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Fold loop ─────────────────────────────────────────────────────────────
    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx+1} / {args.n_folds}  |  mode={args.mode}")
        print(f"{'─'*60}")

        train_ds, val_ds = build_datasets(args, train_df, val_df)

        train_loader = DataLoader(
            train_ds,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = args.num_workers,
            pin_memory  = (device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = (device == "cuda"),
        )

        model = build_model(args).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr           = args.lr,
            weight_decay = args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode      = "min",
            patience  = args.patience,
            factor    = args.lr_factor,
            min_lr    = args.min_lr,
        )

        best_val_acc  = 0.0
        best_ckpt     = os.path.join(mode_dir, f"best_fold{fold_idx+1}.pt")
        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        log_every = max(1, args.epochs // 20)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion,
                scaler, device, args.mode,
            )
            vl_loss, vl_acc = evaluate(
                model, val_loader, criterion, device, args.mode,
            )
            scheduler.step(vl_loss)

            train_losses.append(tr_loss); val_losses.append(vl_loss)
            train_accs.append(tr_acc);   val_accs.append(vl_acc)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), best_ckpt,
                           _use_new_zipfile_serialization=True)

            if epoch % log_every == 0 or epoch == 1:
                lr  = optimizer.param_groups[0]["lr"]
                dt  = time.time() - t0
                print(
                    f"  Epoch {epoch:>4}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                    f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
                    f"lr={lr:.6f} | {dt:.1f}s"
                )

        # ── Save training history CSV (for dashboard curves) ─────────────────
        history_df = pd.DataFrame({
            "epoch":      list(range(1, args.epochs + 1)),
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "train_acc":  train_accs,
            "val_acc":    val_accs,
        })
        history_path = os.path.join(mode_dir, f"history_fold{fold_idx+1}.csv")
        history_df.to_csv(history_path, index=False)

        # ── MC inference on best checkpoint ───────────────────────────────────
        print(f"\n  Loading best checkpoint (val_acc={best_val_acc:.4f})...")
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )

        print(f"  Running MC inference (T={args.mc_passes})...")
        labels, preds, probs, mean_unc = run_mc_inference(
            model, val_loader, device, args.mode, T=args.mc_passes,
        )

        mc_acc = float((labels == preds).mean())
        eer    = compute_multiclass_eer(labels, probs, args.n_classes)

        fold_mc_accs.append(mc_acc)
        fold_eers.append(eer)
        all_labels_global.extend(labels.tolist())
        all_preds_global.extend(preds.tolist())

        # ── Save per-sample predictions CSV (for dashboard uncertainty plot) ──
        # Uncertainty per sample = mean std across classes over T passes
        # We need to rerun inference collecting per-sample uncertainty
        per_sample_unc = _collect_per_sample_uncertainty(
            model, val_loader, device, args.mode, T=args.mc_passes,
        )
        preds_df = pd.DataFrame({
            "true_label":  labels,
            "pred_label":  preds,
            "correct":     (labels == preds).astype(int),
            "uncertainty": per_sample_unc,
            "user":        [label_to_user.get(int(l), str(l)) for l in labels],
        })
        preds_path = os.path.join(mode_dir, f"predictions_fold{fold_idx+1}.csv")
        preds_df.to_csv(preds_path, index=False)

        print(f"\n  ── Fold {fold_idx+1} Results {'─'*26}")
        print(f"  Best val accuracy  (standard) : {best_val_acc:.4f}")
        print(f"  MC accuracy                   : {mc_acc:.4f}")
        print(f"  EER                           : {eer:.4f}")
        print(f"  Mean uncertainty              : {mean_unc:.4f}")

        plot_training_curves(
            train_losses, val_losses,
            train_accs,   val_accs,
            fold_idx,     mode_dir, args.mode,
        )

    # ── Final aggregation ─────────────────────────────────────────────────────
    mean_acc = float(np.mean(fold_mc_accs))
    std_acc  = float(np.std(fold_mc_accs))
    mean_eer = float(np.mean(fold_eers))
    std_eer  = float(np.std(fold_eers))

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {args.mode.upper()} experiment")
    if args.filter != "all":
        print(f"  Filter: {filter_label}")
    print(f"{'='*60}")
    for i, (acc, eer) in enumerate(zip(fold_mc_accs, fold_eers)):
        print(f"  Fold {i+1}:  MC acc={acc:.4f}  EER={eer:.4f}")
    print(f"\n  Mean MC accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean EER         : {mean_eer:.4f} ± {std_eer:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(
        np.array(all_labels_global),
        np.array(all_preds_global),
        label_to_user,
        mode_dir,
        filename="confusion_matrix_all_folds.png",
    )

    # ── Summary file (human readable) ────────────────────────────────────────
    summary_path = os.path.join(mode_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"BioVid Speaker Identification — {args.mode.upper()}\n")
        if args.filter != "all":
            f.write(f"Filter : {filter_label}\n")
        f.write(f"{'─'*40}\n")
        f.write(f"Epochs      : {args.epochs}\n")
        f.write(f"Batch size  : {args.batch_size}\n")
        f.write(f"Feature dim : {args.feature_dim}\n")
        f.write(f"Dropout p   : {args.dropout_p}\n")
        f.write(f"MC passes   : {args.mc_passes}\n")
        if args.mode in ("video", "multimodal"):
            f.write(f"N frames    : {args.n_frames}\n")
        f.write(f"{'─'*40}\n")
        for i, (acc, eer) in enumerate(zip(fold_mc_accs, fold_eers)):
            f.write(f"Fold {i+1}: MC acc={acc:.4f}  EER={eer:.4f}\n")
        f.write(f"{'─'*40}\n")
        f.write(f"Mean MC accuracy : {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"Mean EER         : {mean_eer:.4f} +/- {std_eer:.4f}\n")
    print(f"  📄 Summary saved → {summary_path}")

    # ── metrics.json (machine readable — used by dashboard) ──────────────────
    metrics = {
        "experiment": {
            "mode":        args.mode,
            "paper":       args.paper,
            "filter":      args.filter,
            "arch_label":  arch_label,
            "mode_tag":    mode_tag,
        },
        "hyperparams": {
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "weight_decay":args.weight_decay,
            "dropout_p":   0.2 if args.paper else args.dropout_p,
            "feature_dim": args.feature_dim,
            "mc_passes":   args.mc_passes,
            "n_folds":     args.n_folds,
            "n_frames":    args.n_frames if args.mode in ("video", "multimodal") else None,
            "n_classes":   args.n_classes,
        },
        "folds": [
            {
                "fold":    i + 1,
                "mc_acc":  float(fold_mc_accs[i]),
                "eer":     float(fold_eers[i]),
            }
            for i in range(len(fold_mc_accs))
        ],
        "summary": {
            "mean_mc_acc": float(mean_acc),
            "std_mc_acc":  float(std_acc),
            "mean_eer":    float(mean_eer),
            "std_eer":     float(std_eer),
        },
        "files": {
            "confusion_matrix": "confusion_matrix_all_folds.png",
            "history_csvs":     [f"history_fold{i+1}.csv"  for i in range(args.n_folds)],
            "prediction_csvs":  [f"predictions_fold{i+1}.csv" for i in range(args.n_folds)],
            "curve_pngs":       [f"curves_fold{i+1}.png"   for i in range(args.n_folds)],
        },
    }
    metrics_path = os.path.join(mode_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄 Metrics JSON saved → {metrics_path}")

    # ── Return metrics dict (used by comparison plots) ─────────────────────
    return {
        "mc_acc":     mean_acc,
        "mc_acc_std": std_acc,
        "eer":        mean_eer,
        "eer_std":    std_eer,
    }


if __name__ == "__main__":
    main()