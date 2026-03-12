# BioVid Dashboard — Improved

## Setup

```bash
pip install streamlit plotly pandas numpy
```

## Folder Structure

All dashboard files and the `results/` folder must be in the **same directory**:

```
dashboard/
├── data_loader.py               ← shared module (REQUIRED)
├── overview_dashboard.py
├── modality_dashboard.py
├── audio_architecture_dashboard.py
├── cross_validation_dashboard.py
├── uncertainty_dashboard.py
└── results/
    ├── audio/
    │   ├── metrics.json                    ← required
    │   ├── history_fold1.csv ... fold5.csv ← for training curves
    │   ├── predictions_fold1.csv ... fold5 ← for uncertainty (optional)
    │   ├── confusion_matrix_all_folds.png  ← for images tab
    │   └── curves_fold1.png ... fold5.png  ← for images tab
    ├── audio_paper/   (same files)
    ├── video/         (same files)
    └── multimodal/    (same files)
```

## Running

### Single page
```bash
streamlit run overview_dashboard.py
```

### Multi-page app (recommended)
Create a `pages/` subfolder, move the non-overview dashboards there with numbered prefixes:
```
dashboard/
├── data_loader.py
├── overview_dashboard.py          ← entry point
└── pages/
    ├── 1_Modality.py
    ├── 2_Audio_Architecture.py
    ├── 3_Cross_Validation.py
    └── 4_Uncertainty.py
```
Then run:
```bash
streamlit run overview_dashboard.py
```
Streamlit will auto-detect the `pages/` folder and show a sidebar navigation.

## What Changed vs Original

| Feature | Before | After |
|---|---|---|
| Data source | Hardcoded strings (wrong numbers) | Loaded from `metrics.json` at runtime |
| Audio Paper in CV dashboard | Missing entirely | Included via `build_fold_df()` |
| Training curves | Not shown | Interactive Plotly charts from `history_fold*.csv` |
| All-fold overlay | Not shown | Val accuracy overlay for all 5 folds per experiment |
| Confusion matrices | Not shown | `st.image()` viewer with experiment selector |
| Per-fold curve images | Not shown | Loaded from `curves_fold*.png` |
| Uncertainty data | Estimated only | Real from `predictions_fold*.csv` if available, estimated with notice if not |
| KPI values | Hardcoded | Computed dynamically from loaded data |
| Insight text numbers | Hardcoded strings | Computed from real data |
| Results status | Not shown | Sidebar ✅/❌ per experiment |
| Radar chart | Not present | Added to Overview |
| Efficiency chart | Not present | Accuracy vs param count (log scale) |
| EER reduction chain | Not present | Computed table in Modality dashboard |

## History CSV Column Names

The loader handles both Keras-style and custom naming:

| Canonical name | Accepted input columns |
|---|---|
| `train_loss` | `train_loss`, `loss` |
| `val_loss` | `val_loss`, `validation_loss` |
| `train_acc` | `train_acc`, `train_accuracy`, `accuracy`, `acc` |
| `val_acc` | `val_acc`, `val_accuracy`, `validation_accuracy` |

## Predictions CSV (for uncertainty)

If `predictions_fold*.csv` is present, the loader looks for:
- `true_label`, `predicted_label` → compute standard accuracy
- `mean_uncertainty` / `uncertainty` / `std_uncertainty` → compute mean uncertainty

If absent, Standard Accuracy and Uncertainty fall back to estimated values with a sidebar warning.
