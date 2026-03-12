import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# ── Import data_loader from parent directory ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import (
    build_summary_df, build_fold_df,
    sidebar_results_status, MODEL_COLORS,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BioVid Dashboard - Uncertainty",
    page_icon="🎲",
    layout="wide",
)

# =========================================================
# LOAD DATA
# =========================================================
summary_df = build_summary_df()
fold_df    = build_fold_df()

if summary_df.empty:
    st.title("BioVid Speaker Identification")
    st.markdown("## MC Dropout and Uncertainty Analysis")
    st.error("No results found. Copy `metrics.json` files into the `results/` subfolders.")
    sidebar_results_status()
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### MC Dropout & Uncertainty")
st.sidebar.info(
    "Analyzes the effect of Monte Carlo inference across models. "
    "MC Accuracy = mean over 30 stochastic forward passes per sample."
)

available_models = summary_df["Model"].tolist()
selected_models  = st.sidebar.multiselect(
    "Select models", options=available_models, default=available_models,
)
show_fold_table    = st.sidebar.checkbox("Show fold-level table", value=False)
show_summary_table = st.sidebar.checkbox("Show summary table",    value=True)
sidebar_results_status()

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

filt_sum  = summary_df[summary_df["Model"].isin(selected_models)].copy()
filt_fold = fold_df[fold_df["Model"].isin(selected_models)].copy()

# ── Safe helpers ───────────────────────────────────────────────────────────
def safe_best(df, col, fn="max"):
    s = df[col].dropna()
    if s.empty:
        return None
    return df.loc[s.idxmax() if fn == "max" else s.idxmin()]

best_mc  = safe_best(filt_sum, "MC_Accuracy", "max")
best_eer = safe_best(filt_sum, "EER",         "min")

# =========================================================
# HEADER
# =========================================================
st.title("BioVid Speaker Identification")
st.markdown("## MC Dropout and Uncertainty Analysis")
st.markdown(
    "MC Accuracy = mean over **30 stochastic forward passes**. "
    "Standard Accuracy (single-pass) is not stored in `metrics.json` — "
    "MC Accuracy is the primary reported metric."
)
st.divider()

# =========================================================
# KPI CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    if best_mc is not None:
        st.metric("Best MC Accuracy", f"{best_mc['MC_Accuracy']:.4f}", best_mc["Model"])
with col2:
    if best_eer is not None:
        st.metric("Lowest EER", f"{best_eer['EER']:.4f}", best_eer["Model"])
with col3:
    st.metric("MC Passes", "30", "per sample")
with col4:
    st.metric("Folds", "5", "stratified CV")

st.divider()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["MC Performance", "Fold Details", "Insights"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Mean MC Accuracy by Model")
        fig_acc = px.bar(
            filt_sum, x="Model", y="MC_Accuracy",
            color="Model", error_y="Std",
            color_discrete_map=MODEL_COLORS,
            text=filt_sum["MC_Accuracy"].map(lambda x: f"{x:.4f}"),
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05]), height=480,
            xaxis_title="", yaxis_title="Mean MC Accuracy",
            showlegend=False, margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with right:
        st.subheader("Mean EER by Model")
        fig_eer = px.bar(
            filt_sum, x="Model", y="EER",
            color="Model", error_y="Std_EER",
            color_discrete_map=MODEL_COLORS,
            text=filt_sum["EER"].map(lambda x: f"{x:.4f}"),
        )
        fig_eer.update_traces(textposition="outside")
        fig_eer.update_layout(
            height=480, xaxis_title="", yaxis_title="Mean EER",
            showlegend=False, margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    st.subheader("MC Accuracy vs EER — Efficiency Frontier")
    fig_scat = px.scatter(
        filt_sum, x="EER", y="MC_Accuracy",
        color="Model", text="Model", size="MC_Accuracy",
        color_discrete_map=MODEL_COLORS,
        hover_data={"Parameters": True, "Std": True, "Std_EER": True},
    )
    fig_scat.update_traces(textposition="top center")
    fig_scat.update_layout(
        yaxis=dict(range=[0, 1.05]), height=480,
        xaxis_title="EER (lower = better)",
        yaxis_title="MC Accuracy (higher = better)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_scat.add_annotation(
        x=0, y=1.04, text="← Ideal corner", showarrow=False,
        font=dict(color="green", size=12), xanchor="left"
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    st.subheader("Result Stability (Std Across Folds)")
    left2, right2 = st.columns(2)
    with left2:
        fig_std_acc = px.bar(
            filt_sum, x="Model", y="Std",
            color="Model", color_discrete_map=MODEL_COLORS,
            text=filt_sum["Std"].map(lambda x: f"{x:.4f}"),
            title="Std of MC Accuracy"
        )
        fig_std_acc.update_traces(textposition="outside")
        fig_std_acc.update_layout(
            height=400, showlegend=False, xaxis_title="",
            margin=dict(l=20, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_std_acc, use_container_width=True)

    with right2:
        fig_std_eer = px.bar(
            filt_sum, x="Model", y="Std_EER",
            color="Model", color_discrete_map=MODEL_COLORS,
            text=filt_sum["Std_EER"].map(lambda x: f"{x:.4f}"),
            title="Std of EER"
        )
        fig_std_eer.update_traces(textposition="outside")
        fig_std_eer.update_layout(
            height=400, showlegend=False, xaxis_title="",
            margin=dict(l=20, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_std_eer, use_container_width=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────
with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("MC Accuracy Across Folds")
        fig_fold_acc = px.line(
            filt_fold, x="Fold", y="MC_Accuracy",
            color="Model", markers=True,
            color_discrete_map=MODEL_COLORS,
        )
        fig_fold_acc.update_layout(
            yaxis=dict(range=[0, 1.05]), height=480,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_fold_acc, use_container_width=True)

    with right:
        st.subheader("EER Across Folds")
        fig_fold_eer = px.line(
            filt_fold, x="Fold", y="EER",
            color="Model", markers=True,
            color_discrete_map=MODEL_COLORS,
        )
        fig_fold_eer.update_layout(
            height=480, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_fold_eer, use_container_width=True)

    left3, right3 = st.columns(2)
    with left3:
        st.subheader("MC Accuracy Distribution")
        fig_box_acc = px.box(
            filt_fold, x="Model", y="MC_Accuracy",
            color="Model", points="all",
            color_discrete_map=MODEL_COLORS,
        )
        fig_box_acc.update_layout(
            yaxis=dict(range=[0, 1.05]), height=460,
            showlegend=False, xaxis_title="",
            margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_box_acc, use_container_width=True)

    with right3:
        st.subheader("EER Distribution")
        fig_box_eer = px.box(
            filt_fold, x="Model", y="EER",
            color="Model", points="all",
            color_discrete_map=MODEL_COLORS,
        )
        fig_box_eer.update_layout(
            height=460, showlegend=False, xaxis_title="",
            margin=dict(l=20, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_box_eer, use_container_width=True)

    if show_fold_table:
        st.markdown("### Fold-Level Data")
        disp = filt_fold.copy()
        disp["MC_Accuracy"] = disp["MC_Accuracy"].map(lambda x: f"{x:.4f}")
        disp["EER"]         = disp["EER"].map(lambda x: f"{x:.6f}")
        st.dataframe(disp.drop(columns=["Fold_num"]), use_container_width=True, hide_index=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 1. Modality Dominates
        MC inference builds on top of learned representations.
        The large accuracy gap between Audio and Video/Multimodal
        is a property of the input signal, not the inference strategy.
        """)
    with c2:
        st.markdown("""
        ### 2. Multimodal Is Both Strong and Stable
        **Multimodal Lightweight** achieves the highest MC Accuracy
        and lowest EER — combining modalities reduces both the
        error rate and result variance across folds.
        """)
    with c3:
        st.markdown("""
        ### 3. EER Is the Deployment Metric
        EER progression: Audio (10.6%) → Audio Paper (1.5%) →
        Video (0.39%) → Multimodal (0.19%). A 56× improvement
        from worst to best — accuracy alone understates this gap.
        """)
    st.markdown("---")
    st.markdown("""
    ### Why MC Dropout Matters
    Standard inference uses a single deterministic forward pass.
    With **T=30 stochastic passes**, each producing a different
    probability vector over 43 classes, the final prediction is
    the argmax of their **mean**. This acts as an ensemble and
    reduces prediction variance — at zero additional training cost.

    The benefit is largest when the model has strong feature
    representations but noisy per-sample predictions: exactly
    the case for the video model on a small dataset (650 videos,
    43 classes). The multimodal model, combining both streams,
    achieves the most stable and accurate predictions.
    """)

# =========================================================
# OPTIONAL SUMMARY TABLE
# =========================================================
if show_summary_table:
    st.divider()
    st.subheader("Summary Table")
    disp = filt_sum.copy()
    for col in ["MC_Accuracy", "Std", "EER", "Std_EER"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)