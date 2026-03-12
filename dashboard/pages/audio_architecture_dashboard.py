"""
audio_architecture_dashboard.py — Audio LW vs Audio Paper-Exact.
Isolates the effect of model capacity on audio-only performance.
Dynamic data from metrics.json + training curve comparison.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import data_loader as dl

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BioVid Dashboard — Audio Architecture",
    page_icon="🎧",
    layout="wide",
)

# =========================================================
# LOAD DATA (audio models only)
# =========================================================
AUDIO_MODELS = ["Audio Lightweight", "Audio Paper-Exact"]

summary_df = dl.build_summary_df()
fold_df    = dl.build_fold_df()

audio_summary = summary_df[summary_df["Model"].isin(AUDIO_MODELS)].copy()
audio_folds   = fold_df[fold_df["Model"].isin(AUDIO_MODELS)].copy()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### Audio Architecture Analysis")
st.sidebar.info(
    "Both models operate on **identical Mel Spectrogram inputs**. "
    "Only the architecture differs: Lightweight (~153K params) vs Paper-Exact (~25M params). "
    "The gap shows the effect of model capacity on audio-only performance."
)
dl.sidebar_results_status()

show_table   = st.sidebar.checkbox("Show raw metrics table", value=False)
metric_focus = st.sidebar.selectbox("Highlight metric", ["MC Accuracy", "EER"])

# =========================================================
# HEADER
# =========================================================
st.title("BioVid Speaker Identification")
st.markdown("## Audio Architecture Comparison")
st.markdown(
    "**Audio Lightweight** (GAP, ~153K params) vs **Audio Paper-Exact** (Spata et al. 2025, ~25M params). "
    "Identical input: Mel Spectrogram (128 bins). Identical training protocol."
)

if audio_summary.empty:
    st.error("No results found. Copy `metrics.json` files into `results/audio/` and `results/audio_paper/`.")
    st.stop()

st.divider()

# =========================================================
# KPI CARDS
# =========================================================
lw    = audio_summary[audio_summary["Model"] == "Audio Lightweight"]
paper = audio_summary[audio_summary["Model"] == "Audio Paper-Exact"]

best_acc = audio_summary.loc[audio_summary["MC_Accuracy"].idxmax()]
best_eer = audio_summary.loc[audio_summary["EER"].idxmin()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Best MC Accuracy",
              f"{best_acc['MC_Accuracy']:.4f}",
              best_acc["Model"])
with c2:
    st.metric("Lowest EER",
              f"{best_eer['EER']:.4f}",
              best_eer["Model"])
with c3:
    if not lw.empty and not paper.empty:
        gain = paper.iloc[0]["MC_Accuracy"] - lw.iloc[0]["MC_Accuracy"]
        st.metric("Accuracy Gain (Paper vs LW)", f"+{gain:.4f}", f"~163× more params")
with c4:
    if not lw.empty and not paper.empty:
        eer_red = (lw.iloc[0]["EER"] - paper.iloc[0]["EER"]) / lw.iloc[0]["EER"]
        st.metric("EER Reduction (Paper vs LW)", f"{eer_red:.1%}", "relative improvement")

st.divider()

# =========================================================
# TABS
# =========================================================
tab_perf, tab_cv, tab_curves, tab_table, tab_insights = st.tabs([
    "📊 Performance", "📈 Cross-Validation", "📉 Training Curves", "🗂 Table", "💡 Insights"
])

# ─── TAB 1: PERFORMANCE ──────────────────────────────────────────────────
with tab_perf:
    left, right = st.columns(2)

    with left:
        st.subheader("MC Accuracy by Architecture")
        fig_acc = px.bar(
            audio_summary, x="Model", y="MC_Accuracy",
            color="Model",
            color_discrete_map={
                "Audio Lightweight": "#EF553B",
                "Audio Paper-Exact": "#FF7F0E",
            },
            text=audio_summary["MC_Accuracy"].map(lambda x: f"{x:.4f}"),
            error_y="Std",
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05]),
            xaxis_tickangle=-5,
            margin=dict(l=20, r=20, t=40, b=40), height=480,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with right:
        st.subheader("EER by Architecture")
        fig_eer = px.bar(
            audio_summary, x="Model", y="EER",
            color="Model",
            color_discrete_map={
                "Audio Lightweight": "#EF553B",
                "Audio Paper-Exact": "#FF7F0E",
            },
            text=audio_summary["EER"].map(lambda x: f"{x:.4f}"),
            error_y="Std_EER",
        )
        fig_eer.update_traces(textposition="outside")
        fig_eer.update_layout(
            xaxis_tickangle=-5,
            margin=dict(l=20, r=20, t=40, b=40), height=480,
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    # Grouped bar: acc and eer side by side
    st.subheader("Accuracy vs EER — Side by Side")
    import pandas as pd
    comp_long = pd.melt(
        audio_summary[["Model", "MC_Accuracy", "EER"]],
        id_vars="Model", var_name="Metric", value_name="Value",
    )
    fig_comp = px.bar(
        comp_long, x="Metric", y="Value", color="Model", barmode="group",
        color_discrete_map={
            "Audio Lightweight": "#EF553B",
            "Audio Paper-Exact": "#FF7F0E",
        },
        text=comp_long["Value"].map(lambda x: f"{x:.4f}"),
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=420,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ─── TAB 2: CROSS-VALIDATION ─────────────────────────────────────────────
with tab_cv:
    st.subheader("Fold-Level Accuracy")
    if not audio_folds.empty:
        fig_line = px.line(
            audio_folds, x="Fold", y="MC_Accuracy", color="Model",
            color_discrete_map={
                "Audio Lightweight": "#EF553B",
                "Audio Paper-Exact": "#FF7F0E",
            },
            markers=True,
        )
        fig_line.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20), height=400,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        fig_eer_fold = px.line(
            audio_folds, x="Fold", y="EER", color="Model",
            color_discrete_map={
                "Audio Lightweight": "#EF553B",
                "Audio Paper-Exact": "#FF7F0E",
            },
            markers=True,
        )
        fig_eer_fold.update_layout(
            margin=dict(l=20, r=20, t=40, b=20), height=360,
        )
        st.plotly_chart(fig_eer_fold, use_container_width=True)
    else:
        st.info("No fold data available.")

# ─── TAB 3: TRAINING CURVES ──────────────────────────────────────────────
with tab_curves:
    st.subheader("Training Curves — Both Architectures")
    st.markdown(
        "Compare how the two architectures learn. "
        "Lightweight trains for 200 epochs; Paper-Exact for 100 epochs."
    )

    curve_fold = st.selectbox("Fold", [1, 2, 3, 4, 5], key="arch_fold")

    col_lw, col_paper = st.columns(2)

    for col, model_name in zip([col_lw, col_paper], AUDIO_MODELS):
        with col:
            st.markdown(f"**{model_name}**")
            hist = dl.load_history(model_name, curve_fold)
            if hist is not None:
                color = "#EF553B" if "Lightweight" in model_name else "#FF7F0E"
                ep = hist.get("epoch", range(len(hist)))

                if "train_loss" in hist.columns and "val_loss" in hist.columns:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=ep, y=hist["train_loss"],
                                                  mode="lines", name="Train",
                                                  line=dict(color=color)))
                    fig_loss.add_trace(go.Scatter(x=ep, y=hist["val_loss"],
                                                  mode="lines", name="Val",
                                                  line=dict(color="#636EFA", dash="dash")))
                    fig_loss.update_layout(
                        title="Loss", xaxis_title="Epoch",
                        margin=dict(l=20, r=20, t=40, b=20), height=320,
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                if "train_acc" in hist.columns and "val_acc" in hist.columns:
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Scatter(x=ep, y=hist["train_acc"],
                                               mode="lines", name="Train",
                                               line=dict(color=color)))
                    fig_a.add_trace(go.Scatter(x=ep, y=hist["val_acc"],
                                               mode="lines", name="Val",
                                               line=dict(color="#636EFA", dash="dash")))
                    fig_a.update_layout(
                        title="Accuracy", xaxis_title="Epoch",
                        yaxis=dict(range=[0, 1.05]),
                        margin=dict(l=20, r=20, t=40, b=20), height=320,
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
            else:
                img = dl.get_curves_path(model_name, curve_fold)
                if img:
                    st.image(str(img), use_column_width=True)
                else:
                    st.info(f"No history CSV found for fold {curve_fold}.")

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion Matrices")
    col_cm_lw, col_cm_paper = st.columns(2)
    for col, model_name in zip([col_cm_lw, col_cm_paper], AUDIO_MODELS):
        with col:
            cm_path = dl.get_confusion_matrix_path(model_name)
            if cm_path:
                st.image(str(cm_path), caption=f"{model_name}", use_column_width=True)
            else:
                st.info(f"No confusion matrix found for **{model_name}**.")

# ─── TAB 4: TABLE ────────────────────────────────────────────────────────
with tab_table:
    st.subheader("Audio Architecture Metrics")
    tbl = audio_summary.copy()
    tbl["MC_Accuracy"] = tbl.apply(lambda r: f"{r['MC_Accuracy']:.4f} ± {r['Std']:.4f}", axis=1)
    tbl["EER"] = tbl.apply(lambda r: f"{r['EER']:.4f} ± {r['Std_EER']:.4f}", axis=1)
    tbl = tbl.drop(columns=["Std", "Std_EER", "Modality"])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if not lw.empty and not paper.empty:
        import pandas as pd
        rel = pd.DataFrame({
            "Comparison": [
                "Paper-Exact vs Lightweight — Accuracy Gain",
                "Paper-Exact vs Lightweight — EER Reduction (relative)",
            ],
            "Value": [
                f"+{paper.iloc[0]['MC_Accuracy'] - lw.iloc[0]['MC_Accuracy']:.4f}",
                f"{(lw.iloc[0]['EER'] - paper.iloc[0]['EER']) / lw.iloc[0]['EER']:.1%}",
            ],
        })
        st.markdown("### Relative Improvement")
        st.dataframe(rel, use_container_width=True, hide_index=True)

# ─── TAB 5: INSIGHTS ─────────────────────────────────────────────────────
with tab_insights:
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### 1. Large Performance Gap\n"
            "The Paper-Exact model substantially outperforms Lightweight on audio. "
            "A 163× parameter increase translates into a large, consistent accuracy gain "
            "across all 5 folds."
        )
    with c2:
        st.markdown(
            "### 2. EER Drops Sharply\n"
            "The Paper-Exact architecture has much better score separability, "
            "as reflected by the sharp EER reduction. "
            "The Dense(512) bottleneck enables richer spectro-temporal representations."
        )
    with c3:
        st.markdown(
            "### 3. Audio Is Not Inherently Weak\n"
            "The Lightweight failure on audio is a capacity problem, not an inherent modality problem. "
            "The Paper architecture recovers >80% accuracy from the same Mel Spectrogram inputs."
        )
    st.markdown("---")
    st.markdown(
        "### Interpretation\n"
        "Audio requires greater representational capacity than video to extract reliable "
        "speaker identity cues from BioVid's noisy, short-duration recordings. "
        "The lightweight architecture works well for video (where the lip region is a strong "
        "and clean signal) but is insufficient for audio (where spectro-temporal subtleties "
        "need a deeper, wider projection to separate 43 identities)."
    )

if show_table:
    st.divider()
    st.dataframe(audio_summary, use_container_width=True, hide_index=True)
