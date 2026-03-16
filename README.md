# BioVid Speaker Identification: Multimodal Bayesian Deep Learning

## 1. Project Objective

This repository implements a multimodal biometric authentication and speaker identification system. The primary research objective is to evaluate the discriminative power of visual lip kinematics versus acoustic speech signals in a constrained environment.

Furthermore, the project extends standard deterministic neural networks by integrating Monte Carlo Dropout (MC Dropout) to approximate Bayesian inference. This provides robust epistemic uncertainty quantification alongside classification, making the system suitable for liveness detection and spoof-rejection scenarios.

## 2. Dataset and Preprocessing

The models are trained and evaluated on the **BioVid** dataset:

* **Scope:** 650 videos across 43 unique speakers (average of 15.1 videos per speaker).
* **Visual Stream:** Pre-cropped lip region frames, sampled dynamically to $N=16$ frames per sequence, maintaining a 2:1 aspect ratio (128x64) to preserve spatial integrity.
* **Audio Stream:** 2D Mel-Spectrogram representations extracted via `librosa`, standardized to a fixed temporal window to capture phonetic and prosodic cues.
* **Dynamic Ground Truth Extraction:** To strictly prevent data leakage and task misalignment, the "correct password" for each user is not hardcoded. The ETL pipeline dynamically infers the true password by calculating the statistical mode (highest frequency spoken word) within each individual user's directory.

## 3. Model Architectures & The Efficiency Paradox

A core finding of this project is the demonstration that modality quality can dominate architectural capacity.

1. **Lightweight Unimodal Encoders (~153K parameters):** A custom architecture utilizing 3 Convolutional blocks followed by `AdaptiveAvgPool2d(1)` (Global Average Pooling). This bypasses massive fully-connected layers, acting as a structural regularizer to prevent overfitting on the small 650-sample dataset.
2. **Paper-Exact Audio Baseline (~25M parameters):** A strict architectural replication of the CNNMC model proposed by Spata et al. (2025). This massive network is utilized exclusively to establish a rigorous baseline for the audio-only modality.
3. **Multimodal Late Fusion (~307K parameters):** Independent lightweight encoders process the Mel-spectrograms and the 16-frame kinematic sequences. The resulting latent representations are concatenated at the feature level before passing through a joint classification head.

## 4. Bayesian Inference via MC Dropout

Standard inference yields deterministic outputs that cannot distinguish between data noise and model ignorance. In this pipeline, dropout ($p=0.4$) remains active during the evaluation phase.

For each validation sample, the system executes $T=30$ stochastic forward passes. The final prediction is the argmax of the mean probability vector. The mean standard deviation across the 43-class distributions serves as the predictive uncertainty ($\bar{\sigma}$), providing a mathematical measure of the system's confidence in its authentication decision.

## 5. Experimental Results

The system was evaluated using 5-Fold Stratified Cross-Validation to ensure robust performance metrics without identity leakage.

| Modality / Architecture | Parameter Count | MC Accuracy (T=30) | Equal Error Rate (EER) |
| --- | --- | --- | --- |
| **Audio** (Lightweight GAP) | ~153K | 0.2846 ± 0.039 | 10.63% |
| **Audio** (Spata et al. 2025) | ~25M | 0.8184 ± 0.024 | 1.54% |
| **Video** (Lightweight GAP) | ~153K | 0.8753 ± 0.021 | 0.39% |
| **Multimodal** (Late Fusion) | **~307K** | **0.9400 ± 0.028** | **0.19%** |

*Conclusion:* The 153K-parameter visual model significantly outperforms the 25M-parameter audio baseline. Fusing both modalities yields a highly separable identity distribution, reducing the Equal Error Rate to a near-perfect 0.19%.

## 6. Repository Structure

```text
biovid_project/
├── data/                   # Raw BioVid dataset (git-ignored)
├── src/                    # PyTorch deep learning pipeline
│   ├── model.py            # GAP and CNNMC network topologies
│   ├── train.py            # Training loop with AMP and 5-Fold CV
│   ├── utils.py            # Dynamic labeling, EER calculation, plotting
│   ├── audio_dataset.py    # Mel-spectrogram transformation
│   └── video_dataset.py    # Kinematic frame extraction
├── results/                # Evaluation artifacts and serialized metrics
├── dashboard/              # Streamlit frontend for analytics
│   ├── overview_dashboard.py
    ├── data_loader.py
│   └── pages/              # Multi-page routing architecture
├── docs/                   # Academic documentation
├── requirements.txt        # Python dependencies
└── README.md

```

## 7. Setup and Execution

### Local Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

### Model Training

The training pipeline supports Automatic Mixed Precision (AMP). Due to the parameter scale of the paper baseline, execution on a cloud GPU (e.g., NVIDIA Tesla T4) is recommended for the 25M model.

```bash
# 1. Train Audio (Lightweight)
python src/train.py --mode audio

# 2. Train Audio Baseline (Spata et al. 2025)
python src/train.py --mode audio --paper

# 3. Train Video (Lightweight)
python src/train.py --mode video

# 4. Train Multimodal (Late Fusion)
python src/train.py --mode multimodal

```

### Dashboard Deployment

To visualize the cross-validation metrics, confusion matrices, and the Bayesian uncertainty analysis, launch the local web server:

```bash
cd dashboard
streamlit run overview_dashboard.py

```

*The interactive application will bind to `localhost:8501`.*

## 8. Contributors

* **Nicolò Occhino** 
* **Francesca Calcagno** 
