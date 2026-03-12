# BioVid Speaker Identification
### Multimodal Bayesian Deep Learning — University of Catania, M.Sc. Data Science

---

5-fold stratified cross-validation · 43 speakers · 650 videos · MC Dropout T=30

---

## Project Structure

```
biovid_project/
├── data/                   # ← NOT tracked by Git (too large)
│ 
│
├── src/                    # ← Tracked — all training code
│   ├── model.py            # Lightweight GAP + Paper-Exact architectures
│   ├── train.py            # 5-fold CV training script (all modes)
│   ├── utils.py            # EER, MC Dropout inference, metrics export
│   ├── audio_dataset.py    # Mel Spectrogram pipeline
│   └── video_dataset.py    # Frame sampling pipeline (N=16)
│
├── results/                # ← Partially tracked (see below)
│   ├── audio/              # metrics.json + summary.txt tracked
│   ├── audio_paper/        # heavy CSVs/PNGs in .gitignore
│   ├── video/
│   └── multimodal/
│
├── dashboard/              # ← Tracked — Streamlit dashboard 
│   ├── overview_dashboard.py
│   └── pages/
│       ├── 1_Modality.py
│       ├── 2_Cross_Validation.py
│       ├── 3_Audio_Architecture.py
│       └── 4_Uncertainty.py
│
├── docs/
│   └── BioVid_Project_Documentation.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## What is tracked vs ignored

| Path | Git |
|---|---|
| `src/` | ✅ Fully tracked |
| `dashboard/` | ✅ Fully tracked |
| `docs/` | ✅ Fully tracked |
| `requirements.txt` | ✅ Tracked |
| `results/*/metrics.json` | ✅ Tracked |
| `results/*/summary.txt` | ✅ Tracked |
| `results/*/history_fold*.csv` | ❌ Ignored (heavy) |
| `results/*/predictions_fold*.csv` | ❌ Ignored (heavy) |
| `results/*/curves_fold*.png` | ❌ Ignored (heavy) |
| `data/` | ❌ Ignored (raw video files) |

---

## How to Run Training

```bash
# Install dependencies
pip install -r requirements.txt

# Audio — lightweight (GAP)
python src/train.py --mode audio

# Audio — paper architecture (Spata et al. 2025)
python src/train.py --mode audio --paper

# Video — lightweight
python src/train.py --mode video

# Multimodal — late fusion
python src/train.py --mode multimodal
```

All results are saved to `results/<experiment>/`.

---

## How to Run the Dashboard

```bash
cd dashboard
streamlit run overview_dashboard.py
# Opens at http://localhost:8501
```

---

## Architecture

**Lightweight (GAP) — ~153K params**
- 3× [Conv2D → BatchNorm → ReLU → MaxPool → Dropout]
- Global Average Pooling
- Dense(43) + Softmax

**Paper-Exact (Spata et al. 2025) — ~25M params**
- 4× Conv blocks → Flatten → Dense(512) → Dropout → Dense(43)

**Multimodal Fusion — ~307K params**
- Audio encoder (Lightweight) + Video encoder (Lightweight)
- Late fusion via concatenation → Dense(43)

---

## Bayesian Inference (MC Dropout)

At test time, dropout is kept active and T=30 stochastic forward passes are run per sample. The final prediction is the argmax of the mean over 30 softmax outputs. Uncertainty is estimated as the mean standard deviation of the 30 probability vectors.

---

## Authors

- Nicolò Carmelo Occhino 
- Francesca Calcagno 
