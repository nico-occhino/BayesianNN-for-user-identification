"""
audio_dataset.py — Mel-Spectrogram extraction for the BioVid audio experiment.

Key facts about BioVid audio:
    Duration  : avg 1.77s | min 0.53s | max 3.27s
    Videos    : pre-cropped lip region, 256×128 resolution
    Audio     : extracted from the lip video via ffmpeg

Pipeline: mp4 → ffmpeg → wav → librosa → Mel-Spectrogram (128, T) → [0,1]

Target duration is set to 2.0s to match the dataset's average clip length.
This avoids excessive looping of very short clips (0.53s clips loop ~4×
instead of the previous ~7×). The CNN uses Global Average Pooling so the
exact time dimension T does not need to be fixed.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

# ── Ensure src/ is always importable ─────────────────────────────────────────
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import create_dataframe, get_folds, verify_no_leak


# ─────────────────────────────────────────────────────────────────────────────
# FFMPEG DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Returns path to ffmpeg, searching PATH then common install locations."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
        ),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    raise RuntimeError(
        "\n🚨 ffmpeg not found.\n"
        "Windows : winget install --id Gyan.FFmpeg -e\n"
        "Linux   : sudo apt install ffmpeg\n"
        "Then restart your terminal."
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_to_wav(video_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extracts the audio track from an mp4 using ffmpeg.
    Returns waveform np.ndarray (N,) or empty array on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ffmpeg = _find_ffmpeg()
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path,
             "-ac", "1", "-ar", str(sr), "-vn", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        return y

    except (RuntimeError, subprocess.CalledProcessError) as e:
        if isinstance(e, RuntimeError):
            print(e)
        return np.array([], dtype=np.float32)

    except Exception as e:
        print(f"  ⚠️  Audio extraction error for {video_path}: {e}")
        return np.array([], dtype=np.float32)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Target audio duration matches BioVid average clip length (1.77s).
# Using 2.0s avoids over-looping of short clips while covering most content.
_TARGET_DURATION_S = 2.0
_SR                = 16000
_N_MELS            = 128
_HOP_LENGTH        = 256    # shorter hop → more time bins per second
_N_FFT             = 512


def process_audio(
    video_path: str,
    sr:         int = _SR,
    n_mels:     int = _N_MELS,
    hop_length: int = _HOP_LENGTH,
    n_fft:      int = _N_FFT,
) -> np.ndarray:
    """
    Full pipeline: mp4 → waveform → Mel-Spectrogram → normalised float32.

    Returns:
        np.ndarray (n_mels, T) float32, values in [0, 1]
        Returns zeros on any failure.

    Time dimension T ≈ target_duration * sr / hop_length
                    ≈ 2.0 * 16000 / 256 ≈ 125 frames
    """
    target_samples = int(_TARGET_DURATION_S * sr)

    try:
        y = extract_audio_to_wav(video_path, sr=sr)

        if len(y) == 0:
            return np.zeros((n_mels, int(target_samples / hop_length) + 1),
                            dtype=np.float32)

        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) == 0:
            return np.zeros((n_mels, int(target_samples / hop_length) + 1),
                            dtype=np.float32)

        # Pre-emphasis to boost high frequencies
        y = librosa.effects.preemphasis(y)

        # Loop short clips to reach target length
        if len(y) < target_samples:
            reps = int(np.ceil(target_samples / len(y)))
            y    = np.tile(y, reps)

        # Centre-crop long clips
        start = (len(y) - target_samples) // 2
        y     = y[start: start + target_samples]

        # Mel-Spectrogram
        mel    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length, fmax=sr // 2,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Min-max normalise to [0, 1]
        s_min, s_max = mel_db.min(), mel_db.max()
        if s_max > s_min:
            mel_db = (mel_db - s_min) / (s_max - s_min)
        else:
            mel_db = np.zeros_like(mel_db)

        return mel_db.astype(np.float32)

    except Exception as e:
        print(f"  ⚠️  process_audio failed for {video_path}: {e}")
        return np.zeros((n_mels, int(target_samples / hop_length) + 1),
                        dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SPECAUGMENT  (training-only augmentation)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """
    SpecAugment — frequency and time masking for spectrograms.
    (Park et al., 2019 — https://arxiv.org/abs/1904.08779)

    Applied ONLY to the training set. Validation sees clean spectrograms.
    Cache stores clean spectrograms; augmentation is applied after retrieval
    so every epoch sees a different random mask.
    """

    def __init__(
        self,
        freq_mask_param: int   = 15,
        time_mask_param: int   = 15,
        num_freq_masks:  int   = 2,
        num_time_masks:  int   = 2,
        fill_value:      float = 0.0,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks  = num_freq_masks
        self.num_time_masks  = num_time_masks
        self.fill_value      = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (1, F, T)  →  (1, F, T) with random masks."""
        x = x.clone()
        _, F, T = x.shape

        for _ in range(self.num_freq_masks):
            f  = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            x[:, f0: f0 + f, :] = self.fill_value

        for _ in range(self.num_time_masks):
            t  = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x[:, :, t0: t0 + t] = self.fill_value

        return x


def get_train_transform() -> SpecAugment:
    """Returns the standard SpecAugment transform for training."""
    return SpecAugment(
        freq_mask_param=15,
        time_mask_param=15,
        num_freq_masks=2,
        num_time_masks=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    Returns (spectrogram, label) pairs.

    spectrogram : (1, n_mels, T) float32  — channel-first, values in [0,1]
    label       : torch.long scalar       — integer identity 0..N-1

    Cache: clean spectrograms are cached in RAM after first extraction.
    Augmentation is applied after cache retrieval (fresh mask each epoch).

    fixed_size : if not None, spectrogram is resized to (fixed_size[0], fixed_size[1])
                 Required for PaperAudioModel which needs exactly (128, 128) input.
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        transform:  Optional[nn.Module] = None,
        use_cache:  bool                = True,
        sr:         int                 = _SR,
        n_mels:     int                 = _N_MELS,
        hop_length: int                 = _HOP_LENGTH,
        n_fft:      int                 = _N_FFT,
        fixed_size: Optional[tuple]     = None,   # e.g. (128, 128) for paper model
    ):
        self.paths      = df["path"].values
        self.labels     = df["label"].values
        self.transform  = transform
        self.use_cache  = use_cache
        self.sr         = sr
        self.n_mels     = n_mels
        self.hop_length = hop_length
        self.n_fft      = n_fft
        self.fixed_size = fixed_size
        self.cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cache and idx in self.cache:
            spec = self.cache[idx]
        else:
            arr  = process_audio(
                self.paths[idx],
                sr=self.sr, n_mels=self.n_mels,
                hop_length=self.hop_length, n_fft=self.n_fft,
            )
            spec = torch.from_numpy(arr).unsqueeze(0)   # (1, F, T)

            # Resize to fixed spatial size if required (paper model needs 128×128)
            if self.fixed_size is not None:
                import torch.nn.functional as F_nn
                spec = F_nn.interpolate(
                    spec.unsqueeze(0),               # (1, 1, F, T)
                    size=self.fixed_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)                         # (1, 128, 128)

            if self.use_cache:
                self.cache[idx] = spec

        if self.transform is not None:
            spec = self.transform(spec)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    print("=" * 60)
    print("AUDIO DATASET SELF-TEST")
    print("=" * 60)

    print("\n[0] Checking ffmpeg...")
    ffmpeg = _find_ffmpeg()
    r = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
    print(f"  ✓ {r.stdout.splitlines()[0]}")

    print("\n[1] Building catalog...")
    df, label_to_user = create_dataframe(DATA_DIR)

    print("\n[2] Building folds...")
    folds        = get_folds(df, n_splits=5)
    train_df, val_df = folds[0]
    verify_no_leak(train_df, val_df)

    print("\n[3] Extracting one spectrogram...")
    spec = process_audio(df["path"].iloc[0])
    print(f"  Shape : {spec.shape}  range [{spec.min():.3f}, {spec.max():.3f}]")
    assert not (spec == 0).all(), "🚨 All zeros — audio extraction failed"

    print("\n[4] SpecAugment visual check...")
    aug         = get_train_transform()
    spec_tensor = torch.from_numpy(spec).unsqueeze(0)
    fig, axes   = plt.subplots(1, 4, figsize=(16, 3))
    axes[0].imshow(spec_tensor[0].numpy(), aspect="auto",
                   origin="lower", cmap="magma")
    axes[0].set_title("Original"); axes[0].axis("off")
    for i in range(1, 4):
        a = aug(spec_tensor)
        axes[i].imshow(a[0].numpy(), aspect="auto", origin="lower", cmap="magma")
        axes[i].set_title(f"Aug {i}"); axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("specaugment_check.png", dpi=120)
    plt.close()
    print("  📊 Saved → specaugment_check.png")

    print("\n[5] DataLoader smoke test...")
    train_ds = AudioDataset(train_df, transform=get_train_transform(),
                            use_cache=False)
    val_ds   = AudioDataset(val_df,   transform=None, use_cache=False)
    tx, ty   = next(iter(DataLoader(train_ds, batch_size=4, num_workers=0)))
    vx, vy   = next(iter(DataLoader(val_ds,   batch_size=4, num_workers=0)))
    print(f"  Train : {tx.shape}  Val : {vx.shape}")

    print("\n[6] Augmentation variance check...")
    ds     = AudioDataset(train_df, transform=get_train_transform(),
                          use_cache=True)
    s1, _  = ds[0]
    s2, _  = ds[0]
    diff   = (s1 - s2).abs().mean().item()
    assert diff > 0, "🚨 Augmentation not random"
    print(f"  Mean diff between calls: {diff:.6f} ✓")
    print("\n✓ Audio dataset self-test PASSED.")