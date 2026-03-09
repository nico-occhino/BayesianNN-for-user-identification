"""
model.py — CNN + MC Dropout models for the BioVid speaker identification project.

Three models share the same CNNEncoder backbone:

    AudioModel     : Mel-Spectrogram (1, F, T)       → identity logits
    VideoModel     : lip frames (N, 1, H, W)          → identity logits
    FusionModel    : audio + video simultaneously     → identity logits

Fusion strategy — Late Feature Fusion:
    audio_feat  = AudioEncoder(spectrogram)     # (B, feature_dim)
    video_feat  = VideoEncoder(frames)          # (B, feature_dim)
    fused       = concat([audio_feat, video_feat])  # (B, 2 * feature_dim)
    logits      = ClassificationHead(fused)

    The two encoders are trained jointly end-to-end from scratch.
    Each encoder learns to produce complementary features because
    the shared classification head must reconcile both modalities.

    Late fusion (at feature level, not decision level) was chosen over:
    - Early fusion  : would require fixed-size inputs for both modalities
    - Score fusion  : would require pre-trained unimodal models
    Late fusion trains everything in one pass and is easier to analyse.

CNNEncoder design note:
    AdaptiveAvgPool2d(1) (Global Average Pooling) is used instead of Flatten
    so the encoder accepts any spatial resolution.
    This lets the same encoder handle:
        AudioModel : (1, 128, ~125) spectrograms
        VideoModel : (1, 64, 128) lip frames (H=64, W=128)
    without any code changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# MC DROPOUT
# ─────────────────────────────────────────────────────────────────────────────

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout — stays active during eval() mode.
    (Gal & Ghahramani, 2016 — https://arxiv.org/abs/1506.02142)

    Standard nn.Dropout is disabled by model.eval(). This layer
    forces training=True so dropout remains stochastic at inference time,
    enabling T forward passes to produce a distribution of predictions.
    """

    def __init__(self, p: float = 0.4):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0,1). Got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)

    def extra_repr(self) -> str:
        return f"p={self.p}, always_active=True"


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool2d (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# CNN ENCODER  (shared backbone)
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Three-block CNN with Global Average Pooling.

    Accepts any single-channel 2D input regardless of spatial size:
        Audio spectrograms : (B, 1, 128, ~125)
        Lip frames         : (B, 1, 64,  128)

    Architecture:
        Block1 : (B, 1,   H,   W)   → (B, 32,  H/2,  W/2)
        Block2 : (B, 32,  H/2, W/2) → (B, 64,  H/4,  W/4)
        Block3 : (B, 64,  H/4, W/4) → (B, 128, H/8,  W/8)
        GAP    : (B, 128, H/8, W/8) → (B, 128, 1,    1)
        Linear : (B, 128)           → (B, feature_dim)
        ReLU
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32,          64),
            ConvBlock(64,          128),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)         # (B, 128, h, w)
        x = self.gap(x)            # (B, 128,  1, 1)
        x = x.flatten(1)           # (B, 128)
        x = self.projection(x)     # (B, feature_dim)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    MCDropout → Linear → ReLU → MCDropout → Linear (logits)

    No Softmax: CrossEntropyLoss expects raw logits during training.
    Softmax is applied in mc_predict() for probability averaging.
    """

    def __init__(
        self,
        in_dim:    int   = 128,
        hidden:    int   = 256,
        n_classes: int   = 43,
        p:         float = 0.4,
    ):
        super().__init__()
        self.head = nn.Sequential(
            MCDropout(p=p),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            MCDropout(p=p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT ARCHITECTURE  (Spata et al., EURASIP 2025)
# ─────────────────────────────────────────────────────────────────────────────

class PaperCNNEncoder(nn.Module):
    """
    Exact replication of the CNN encoder from the published paper.

    From Table 1 and Section 4.3:
        Input  : (B, 1, 128, 128)
        Block1 : Conv(64)  → BN → ReLU → MaxPool(2×2) → (B, 64,  64, 64)
        Block2 : Conv(128) → BN → ReLU → MaxPool(2×2) → (B, 128, 32, 32)
        Block3 : Conv(256) → BN → ReLU → MaxPool(2×2) → (B, 256, 16, 16)
        Block4 : Conv(512) → BN → ReLU → MaxPool(2×2) → (B, 512,  8,  8)
        Flatten                                         → (B, 32768)
        Dense(512) → ReLU                               → (B, 512)

    Parameter count: ~25M  (dominated by Flatten→Dense(512): 32768×512 = 16.7M)
    This is 160× more parameters than our lightweight GAP model.

    IMPORTANT: Requires fixed (128, 128) spatial input. Unlike CNNEncoder
    which uses AdaptiveAvgPool2d, this Flatten layer breaks for other sizes.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 64),    # 128 → 64
            ConvBlock(64,          128),   # 64  → 32
            ConvBlock(128,         256),   # 32  → 16
            ConvBlock(256,         512),   # 16  → 8
        )
        # 512 × 8 × 8 = 32,768 after 4 halvings from 128×128
        self.flatten    = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)      # (B, 512, 8, 8)
        x = self.flatten(x)     # (B, 32768)
        x = self.projection(x)  # (B, 512)
        return x


class PaperAudioModel(nn.Module):
    """
    Full replication of the CNNMC model from Spata et al. (EURASIP 2025).

    Hyperparameters from Table 1:
        Input shape  : (128, 128, 1)  → PyTorch: (B, 1, 128, 128)
        Hidden layers: 4 Conv blocks
        Dropout rate : 0.2
        Optimizer    : Adam, lr=0.0001
        Epochs       : 100
        Batch size   : 32
        Num classes  : 43
        MC passes    : 30

    This model achieves 93.27% accuracy and EER=0.030 on BioVid (paper result).
    Use this for direct comparison with the paper's reported numbers.

    NOTE: Input spectrograms must be resized to exactly (128, 128) for the
    Flatten layer to produce the correct 32,768-dim vector. The audio pipeline
    should pad/crop time axis to 128 frames when using this model.
    """

    def __init__(
        self,
        n_classes:  int   = 43,
        dropout_p:  float = 0.2,   # paper uses 0.2, not 0.4
    ):
        super().__init__()
        self.encoder = PaperCNNEncoder(in_channels=1)
        self.head    = ClassificationHead(
            in_dim    = 512,
            hidden    = 512,
            n_classes = n_classes,
            p         = dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, 128, 128) → logits : (B, n_classes)"""
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — AUDIO MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AudioModel(nn.Module):
    """
    Speaker identification from Mel-Spectrogram images.

    Input  : (B, 1, F, T)  — Mel-Spectrogram, variable T
    Output : (B, n_classes) — logits
    """

    def __init__(
        self,
        feature_dim: int   = 128,
        hidden:      int   = 256,
        n_classes:   int   = 43,
        p:           float = 0.4,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1, feature_dim=feature_dim)
        self.head    = ClassificationHead(feature_dim, hidden, n_classes, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns feature vector without classification head."""
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — VIDEO MODEL
# ─────────────────────────────────────────────────────────────────────────────

class VideoModel(nn.Module):
    """
    Speaker identification from lip video frames via mean-pooled CNN features.

    Input  : (B, N, 1, H, W)  — N uniformly sampled frames, H=64 W=128
    Output : (B, n_classes)   — logits

    The CNN encoder is shared across all N frames (parameter efficient).
    Mean pooling collapses the temporal dimension without adding parameters.
    """

    def __init__(
        self,
        feature_dim: int   = 128,
        hidden:      int   = 256,
        n_classes:   int   = 43,
        p:           float = 0.4,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1, feature_dim=feature_dim)
        self.head    = ClassificationHead(feature_dim, hidden, n_classes, p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mean-pooled feature vector (B, feature_dim)."""
        B, N, C, H, W = x.shape
        x    = x.view(B * N, C, H, W)      # (B*N, 1, H, W)
        feat = self.encoder(x)              # (B*N, feature_dim)
        feat = feat.view(B, N, -1)          # (B, N, feature_dim)
        return feat.mean(dim=1)             # (B, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 — MULTIMODAL FUSION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    """
    Late-feature fusion of audio and video modalities.

    Architecture:
        audio_spec   (B, 1, F, T)      → AudioEncoder  → (B, feature_dim)
        video_frames (B, N, 1, H, W)   → VideoEncoder  → (B, feature_dim)
                                                           ↓ concat
                                                     (B, 2 * feature_dim)
                                                           ↓
                                           ClassificationHead
                                                           ↓
                                                  (B, n_classes)

    Both encoders are trained jointly. The shared head must reconcile
    both modalities, encouraging complementary feature learning.

    Why late fusion over early fusion:
        - No need for fixed-size aligned inputs
        - Each modality keeps its own temporal structure
        - Easier to ablate (disable one modality at inference)
        - More interpretable: we can inspect each encoder's features

    Args:
        feature_dim : output dim for each encoder (default 128)
                      fused representation = 2 × feature_dim = 256
        hidden      : hidden dim in classification head (default 512)
        n_classes   : number of identity classes (default 43)
        p           : MC Dropout probability (default 0.4)
    """

    def __init__(
        self,
        feature_dim: int   = 128,
        hidden:      int   = 512,
        n_classes:   int   = 43,
        p:           float = 0.4,
    ):
        super().__init__()
        self.audio_encoder = CNNEncoder(in_channels=1,
                                        feature_dim=feature_dim)
        self.video_encoder = CNNEncoder(in_channels=1,
                                        feature_dim=feature_dim)
        # Head takes concatenated features: 2 × feature_dim
        self.head = ClassificationHead(
            in_dim    = 2 * feature_dim,
            hidden    = hidden,
            n_classes = n_classes,
            p         = p,
        )

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, 1, F, T) → (B, feature_dim)"""
        return self.audio_encoder(audio)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """(B, N, 1, H, W) → (B, feature_dim)"""
        B, N, C, H, W = video.shape
        x    = video.view(B * N, C, H, W)
        feat = self.video_encoder(x)
        feat = feat.view(B, N, -1)
        return feat.mean(dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio : (B, 1, F, T)
            video : (B, N, 1, H, W)
        Returns:
            logits : (B, n_classes)
        """
        a_feat = self.encode_audio(audio)             # (B, feature_dim)
        v_feat = self.encode_video(video)             # (B, feature_dim)
        fused  = torch.cat([a_feat, v_feat], dim=1)  # (B, 2*feature_dim)
        return self.head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# MC INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(
    model:  nn.Module,
    inputs: Tuple,
    T:      int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo inference: T stochastic forward passes with dropout active.

    Args:
        model  : AudioModel | VideoModel | FusionModel (on device)
        inputs : tuple of input tensors (already on device)
                   AudioModel  → (spectrogram,)
                   VideoModel  → (frames,)
                   FusionModel → (audio_spec, video_frames)
        T      : number of forward passes (default 30)

    Returns:
        mean_probs  : (B, n_classes) — averaged softmax probabilities
        uncertainty : (B,)           — mean std-dev across classes
                                       higher = less confident
    """
    model.eval()   # freezes BatchNorm stats; MCDropout stays active

    all_probs = []
    for _ in range(T):
        logits = model(*inputs)
        probs  = F.softmax(logits, dim=-1)
        all_probs.append(probs.unsqueeze(0))

    all_probs   = torch.cat(all_probs, dim=0)   # (T, B, n_classes)
    mean_probs  = all_probs.mean(dim=0)          # (B, n_classes)
    uncertainty = all_probs.std(dim=0).mean(dim=-1)  # (B,)

    return mean_probs, uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    B           = 4
    N           = 16
    N_CLASSES   = 43
    FEATURE_DIM = 128

    print("=" * 60)
    print(f"MODEL SELF-TEST  |  device={DEVICE}")
    print("=" * 60)

    # ── MCDropout always-on check ─────────────────────────────────────────────
    print("\n[1] MCDropout always-active check...")
    drop = MCDropout(p=0.5)
    x    = torch.ones(200, 200)
    drop.eval()
    zeros = (drop(x) == 0).float().mean().item()
    assert zeros > 0.3, "🚨 MCDropout off during eval!"
    print(f"  ✓ {zeros:.1%} neurons dropped during eval() — correct.")

    # ── AudioModel ────────────────────────────────────────────────────────────
    print("\n[2] AudioModel...")
    m    = AudioModel(FEATURE_DIM, n_classes=N_CLASSES).to(DEVICE)
    x    = torch.randn(B, 1, 128, 125).to(DEVICE)
    out  = m(x)
    assert out.shape == (B, N_CLASSES)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ✓ output {list(out.shape)}  |  {params:,} params")

    # ── VideoModel ────────────────────────────────────────────────────────────
    print("\n[3] VideoModel (H=64, W=128)...")
    m    = VideoModel(FEATURE_DIM, n_classes=N_CLASSES).to(DEVICE)
    x    = torch.randn(B, N, 1, 64, 128).to(DEVICE)
    out  = m(x)
    assert out.shape == (B, N_CLASSES)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ✓ output {list(out.shape)}  |  {params:,} params")

    # ── FusionModel ───────────────────────────────────────────────────────────
    print("\n[4] FusionModel (audio + video)...")
    m     = FusionModel(FEATURE_DIM, n_classes=N_CLASSES).to(DEVICE)
    audio = torch.randn(B, 1, 128, 125).to(DEVICE)
    video = torch.randn(B, N, 1, 64, 128).to(DEVICE)
    out   = m(audio, video)
    assert out.shape == (B, N_CLASSES)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ✓ output {list(out.shape)}  |  {params:,} params")
    print(f"  Audio encoder  : {sum(p.numel() for p in m.audio_encoder.parameters()):,} params")
    print(f"  Video encoder  : {sum(p.numel() for p in m.video_encoder.parameters()):,} params")
    print(f"  Fusion head    : {sum(p.numel() for p in m.head.parameters()):,} params")

    # ── MC inference ──────────────────────────────────────────────────────────
    print("\n[5] MC inference (T=30)...")
    probs, unc = mc_predict(m, (audio, video), T=30)
    assert probs.shape == (B, N_CLASSES)
    print(f"  ✓ mean_probs {list(probs.shape)}  uncertainty {unc.tolist()}")

    # ── Stochastic variation check ────────────────────────────────────────────
    print("\n[6] Verifying stochastic variation across passes...")
    m.eval()
    p1 = F.softmax(m(audio, video), dim=-1)
    p2 = F.softmax(m(audio, video), dim=-1)
    diff = (p1 - p2).abs().mean().item()
    assert diff > 0, "🚨 MC passes are identical — MCDropout not working!"
    print(f"  ✓ mean abs diff between passes: {diff:.6f}")

    print("\n✓ Model self-test PASSED.")