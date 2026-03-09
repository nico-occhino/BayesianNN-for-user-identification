"""
video_dataset.py — Frame extraction for the BioVid video experiment.

BioVid video facts (from Metadata.json):
    Resolution : 256×128 px (width × height) — already cropped to lip region
    FPS        : 24-60 (variable across devices)
    Duration   : avg 1.77s | min 0.53s | max 3.27s
    Frames     : avg ~45 frames per clip at 25fps

Critical fix vs previous version:
    Previous : size=(128, 128) — squashed the lip region into a square,
               destroying the 2:1 aspect ratio and losing spatial information.
    Current  : size=(128, 64)  — preserves aspect ratio (W=128, H=64),
               consistent with the 256×128 source resolution.

    CNN input is now (1, 64, 128) per frame instead of (1, 128, 128).
    The CNNEncoder uses AdaptiveAvgPool2d(1) so it handles any spatial size.
"""

import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

# ── Ensure src/ is always importable ─────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import create_dataframe, get_folds, verify_no_leak


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (match source resolution aspect ratio)
# ─────────────────────────────────────────────────────────────────────────────

# Source resolution is 256×128 (2:1 ratio).
# We downsample by 2× to keep computation light: 128×64.
FRAME_W = 128   # width
FRAME_H = 64    # height
FRAME_SIZE = (FRAME_W, FRAME_H)   # (width, height) for cv2.resize


# ─────────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    n_frames:   int              = 16,
    size:       Tuple[int, int]  = FRAME_SIZE,
) -> np.ndarray:
    """
    Uniformly samples n_frames from a lip video.

    Pipeline:
        1. Open video with OpenCV
        2. Compute n_frames evenly-spaced frame indices across duration
        3. For each index: seek → read → grayscale → resize to (W, H)
        4. Short clips (< n_frames): tile available frames to reach n_frames
        5. Normalise pixels to [0, 1]
        6. Add channel dimension

    Args:
        video_path : path to .mp4 file
        n_frames   : number of frames to sample (default 16)
        size       : (width, height) for resize — default (128, 64)

    Returns:
        np.ndarray (n_frames, 1, H, W) float32
        Returns zero array on any failure.
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Video reports 0 frames: {video_path}")

        # ── Uniform sampling ──────────────────────────────────────────────────
        if total >= n_frames:
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
        else:
            indices = np.arange(total)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret or frame is None:
                frames.append(np.zeros((size[1], size[0]), dtype=np.float32))
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.resize takes (width, height)
            gray  = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32))

        cap.release()

        frames = np.stack(frames, axis=0)   # (available, H, W)

        # ── Tile short clips ──────────────────────────────────────────────────
        if frames.shape[0] < n_frames:
            reps   = int(np.ceil(n_frames / frames.shape[0]))
            frames = np.tile(frames, (reps, 1, 1))

        frames = frames[:n_frames]          # (n_frames, H, W)
        frames = frames / 255.0             # normalise to [0, 1]
        frames = frames[:, np.newaxis, :, :]  # (n_frames, 1, H, W)

        return frames.astype(np.float32)

    except Exception as e:
        print(f"  ⚠️  Frame extraction failed for {video_path}: {e}")
        return np.zeros((n_frames, 1, size[1], size[0]), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class VideoDataset(Dataset):
    """
    Returns (frames, label) pairs.

    frames : (N, 1, H, W) float32  — N uniformly sampled lip frames
    label  : torch.long scalar     — integer identity 0..N-1

    Default frame size: (1, 64, 128) per frame — preserves lip crop ratio.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        n_frames:  int               = 16,
        size:      Tuple[int, int]   = FRAME_SIZE,
        transform                    = None,
        use_cache: bool              = True,
    ):
        self.paths     = df["path"].values
        self.labels    = df["label"].values
        self.n_frames  = n_frames
        self.size      = size
        self.transform = transform
        self.use_cache = use_cache
        self.cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cache and idx in self.cache:
            frames = self.cache[idx]
        else:
            arr    = extract_frames(self.paths[idx],
                                    n_frames=self.n_frames, size=self.size)
            frames = torch.from_numpy(arr)
            if self.use_cache:
                self.cache[idx] = frames

        if self.transform:
            frames = self.transform(frames)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label


# ─────────────────────────────────────────────────────────────────────────────
# MULTIMODAL PAIRED DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """
    Returns (audio_spec, video_frames, label) triples for fusion training.

    Each sample is a single video file from which both modalities are derived:
        audio_spec   : (1, n_mels, T)     float32  — Mel-Spectrogram
        video_frames : (n_frames, 1, H, W) float32  — lip frames

    The audio and video caches are independent and both held in RAM.
    SpecAugment is applied to audio_spec when transform_audio is set.
    """

    def __init__(
        self,
        df:               pd.DataFrame,
        n_frames:         int               = 16,
        frame_size:       Tuple[int, int]   = FRAME_SIZE,
        transform_audio                     = None,
        use_cache:        bool              = True,
    ):
        # Lazy import to avoid circular dependency
        from audio_dataset import process_audio, _N_MELS, _HOP_LENGTH, _N_FFT, _SR

        self.paths           = df["path"].values
        self.labels          = df["label"].values
        self.n_frames        = n_frames
        self.frame_size      = frame_size
        self.transform_audio = transform_audio
        self.use_cache       = use_cache

        self._process_audio = process_audio
        self._n_mels        = _N_MELS
        self._hop_length    = _HOP_LENGTH
        self._n_fft         = _N_FFT
        self._sr            = _SR

        self.audio_cache: Dict[int, torch.Tensor] = {}
        self.video_cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── Audio ─────────────────────────────────────────────────────────────
        if self.use_cache and idx in self.audio_cache:
            audio = self.audio_cache[idx]
        else:
            arr   = self._process_audio(
                self.paths[idx],
                sr=self._sr, n_mels=self._n_mels,
                hop_length=self._hop_length, n_fft=self._n_fft,
            )
            audio = torch.from_numpy(arr).unsqueeze(0)  # (1, F, T)
            if self.use_cache:
                self.audio_cache[idx] = audio

        if self.transform_audio is not None:
            audio = self.transform_audio(audio)

        # ── Video ─────────────────────────────────────────────────────────────
        if self.use_cache and idx in self.video_cache:
            video = self.video_cache[idx]
        else:
            arr   = extract_frames(
                self.paths[idx],
                n_frames=self.n_frames, size=self.frame_size,
            )
            video = torch.from_numpy(arr)
            if self.use_cache:
                self.video_cache[idx] = video

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio, video, label


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    N_FRAMES = 16

    print("=" * 60)
    print("VIDEO DATASET SELF-TEST")
    print("=" * 60)

    print("\n[1] Building catalog...")
    df, label_to_user = create_dataframe(DATA_DIR)

    print("\n[2] Inspecting one video...")
    cap   = cv2.VideoCapture(df["path"].iloc[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  Source  : {w}×{h}px  |  {fps:.1f}fps  |  {total} frames")
    print(f"  Resized : {FRAME_W}×{FRAME_H}px  (aspect ratio preserved)")

    print("\n[3] Building folds...")
    folds            = get_folds(df, n_splits=5)
    train_df, val_df = folds[0]
    verify_no_leak(train_df, val_df)

    print("\n[4] VideoDataset smoke test...")
    ds      = VideoDataset(train_df, n_frames=N_FRAMES, use_cache=False)
    dl      = DataLoader(ds, batch_size=2, num_workers=0)
    bx, by  = next(iter(dl))
    print(f"  Batch shape : {bx.shape}   expect (2, {N_FRAMES}, 1, {FRAME_H}, {FRAME_W})")
    print(f"  Range       : [{bx.min():.3f}, {bx.max():.3f}]")

    print("\n[5] MultimodalDataset smoke test...")
    from audio_dataset import get_train_transform
    mm_ds  = MultimodalDataset(train_df, n_frames=N_FRAMES,
                               transform_audio=get_train_transform(),
                               use_cache=False)
    mm_dl  = DataLoader(mm_ds, batch_size=2, num_workers=0)
    ax, vx, ly = next(iter(mm_dl))
    print(f"  Audio  shape : {ax.shape}")
    print(f"  Video  shape : {vx.shape}")
    print(f"  Labels       : {ly.tolist()}")

    print("\n[6] Visualising frames...")
    frames = bx[0]   # (N, 1, H, W)
    cols   = min(N_FRAMES, 8)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
    for i in range(N_FRAMES):
        r, c = i // cols, i % cols
        axes[r, c].imshow(frames[i, 0].numpy(), cmap="gray", vmin=0, vmax=1,
                          aspect="auto")
        axes[r, c].set_title(f"f{i}", fontsize=7)
        axes[r, c].axis("off")
    plt.suptitle(f"User: {label_to_user[by[0].item()]} — {N_FRAMES} frames "
                 f"({FRAME_W}×{FRAME_H})")
    plt.tight_layout()
    plt.savefig("sample_frames.png", dpi=120)
    plt.close()
    print("  📊 Saved → sample_frames.png")
    print("\n✓ Video dataset self-test PASSED.")