"""
cross_validation_dashboard.py — Fold-level robustness analysis.
Now includes ALL FOUR experiments (Audio Paper was previously missing).
Dynamic data from metrics.json + training curve viewer from history CSVs.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import data_loader as dl

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BioVid Dashboard — Cross-Validation",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# LOAD DATA
# =========================================================
summary_df = dl.build_summary_df()
fold_df    = dl.build_fold_df()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### Cross-Validation Robustness")
st.sidebar.info(
    "Fold-by-fold breakdown for all 4 models. "
    "Confirms that the performance ranking is robust across splits, "
    "not driven by a single favorable partition."
)
dl.sidebar_results_status()

available_models = fold_df["Model"].unique().tolist() if not fold_df.empty else []
selected_models = st.sidebar.multiselect(
    "Select models", options=available_models, default=available_models
)
show_fold_table   = st.sidebar.checkbox("Show fold-level table", value=False)
metric_choice     = st.sidebar.selectbox("Highlight metric", ["MC Accuracy", "EER"])

# =========================================================
# HEADER
# =========================================================
st.title("BioVid Speaker Identification")
st.markdown("## Cross-Validation Robustness")
st.markdown(
    "Fold-by-fold analysis of **all four models** under "
    "**5-fold stratified cross-validation** — 43 speakers, 650 videos."
)

if fold_df.empty:
    st.error("No results found. Copy `metrics.json` files into the `results/` subfolders.")
    st.stop()

st.divider()

# =========================================================
# KPI CARDS
# =========================================================
filt_sum = summary_df[summary_df["Model"].isin(selected_models)]
if filt_sum.empty:
    filt_sum = summary_df

best_acc    = filt_sum.loc[filt_sum["MC_Accuracy"].idxmax()]
most_stable = filt_sum.loc[filt_sum["Std"].idxmin()]
best_eer    = filt_sum.loc[filt_sum["EER"].idxmin()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Best Mean Accuracy",
              f"{best_acc['MC_Accuracy']:.4f}",
              best_acc["Model"])
with c2:
    st.metric("Most Stable (lowest Std)",
              f"± {most_stable['Std']:.4f}",
              most_stable["Model"])
with c3:
    st.metric("Lowest Mean EER",
              f"{best_eer['EER']:.4f}",
              best_eer["Model"])
with c4:
    st.metric("Validation Folds", "5", "Stratified CV")

st.divider()

# =========================================================
# TABS
# =========================================================
tab_fold, tab_curves, tab_summary, tab_insights = st.tabs([
    "📊 Fold Performance", "📉 Training Curves", "🗂 Summary Table", "💡 Insights"
])

filt_fold = fold_df[fold_df["Model"].isin(selected_models)]

# ─── TAB 1: FOLD PERFORMANCE ─────────────────────────────────────────────
with tab_fold:
    left, right = st.columns(2)

    with left:
        st.subheader("MC Accuracy Across Folds")
        fig_acc = px.line(
            filt_fold, x="Fold", y="MC_Accuracy", color="Model",
            color_discrete_map=dl.MODEL_COLORS, markers=True,
        )
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20), height=460,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with right:
        st.subheader("EER Across Folds")
        fig_eer = px.line(
            filt_fold, x="Fold", y="EER", color="Model",
            color_discrete_map=dl.MODEL_COLORS, markers=True,
        )
        fig_eer.update_layout(
            yaxis_title="EER",
            margin=dict(l=20, r=20, t=40, b=20), height=460,
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    st.subheader("Accuracy Distribution by Model (box plot)")
    fig_box = px.box(
        filt_fold, x="Model", y="MC_Accuracy",
        color="Model", points="all",
        color_discrete_map=dl.MODEL_COLORS,
    )
    fig_box.update_layout(
        yaxis=dict(range=[0, 1.05]), showlegend=False,
        xaxis_tickangle=-15,
        margin=dict(l=20, r=20, t=40, b=60), height=440,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("EER Distribution by Model (box plot)")
    fig_box_eer = px.box(
        filt_fold, x="Model", y="EER",
        color="Model", points="all",
        color_discrete_map=dl.MODEL_COLORS,
    )
    fig_box_eer.update_layout(
        showlegend=False, xaxis_tickangle=-15,
        margin=dict(l=20, r=20, t=40, b=60), height=400,
    )
    st.plotly_chart(fig_box_eer, use_container_width=True)

    if show_fold_table:
        st.markdown("### Fold-Level Table")
        tbl = filt_fold.copy()
        tbl["MC_Accuracy"] = tbl["MC_Accuracy"].map(lambda x: f"{x:.4f}")
        tbl["EER"] = tbl["EER"].map(lambda x: f"{x:.6f}")
        st.dataframe(tbl.drop(columns=["Fold_num", "Modality"]), use_container_width=True, hide_index=True)

# ─── TAB 2: TRAINING CURVES ──────────────────────────────────────────────
with tab_curves:
    st.subheader("Training History — Loss & Accuracy per Fold")
    st.markdown(
        "Select an experiment and fold to inspect the training / validation curves. "
        "Files are loaded from `results/<experiment>/history_foldN.csv`."
    )

    col_model, col_fold = st.columns([2, 1])
    with col_model:
        curve_model = st.selectbox(
            "Experiment",
            options=dl.MODEL_NAMES,
            key="cv_curve_model",
        )
    with col_fold:
        curve_fold = st.selectbox(
            "Fold",
            options=[1, 2, 3, 4, 5],
            key="cv_curve_fold",
        )

    hist_df = dl.load_history(curve_model, curve_fold)

    if hist_df is not None:
        cl, cr = st.columns(2)

        with cl:
            st.markdown(f"**Loss — {curve_model} / Fold {curve_fold}**")
            if "train_loss" in hist_df.columns and "val_loss" in hist_df.columns:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=hist_df.get("epoch", range(len(hist_df))),
                    y=hist_df["train_loss"],
                    mode="lines", name="Train Loss",
                    line=dict(color="#636EFA"),
                ))
                fig_loss.add_trace(go.Scatter(
                    x=hist_df.get("epoch", range(len(hist_df))),
                    y=hist_df["val_loss"],
                    mode="lines", name="Val Loss",
                    line=dict(color="#EF553B", dash="dash"),
                ))
                fig_loss.update_layout(
                    xaxis_title="Epoch", yaxis_title="Loss",
                    margin=dict(l=20, r=20, t=30, b=20), height=380,
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("Loss columns not found in CSV.")

        with cr:
            st.markdown(f"**Accuracy — {curve_model} / Fold {curve_fold}**")
            if "train_acc" in hist_df.columns and "val_acc" in hist_df.columns:
                fig_acc_c = go.Figure()
                fig_acc_c.add_trace(go.Scatter(
                    x=hist_df.get("epoch", range(len(hist_df))),
                    y=hist_df["train_acc"],
                    mode="lines", name="Train Acc",
                    line=dict(color="#00CC96"),
                ))
                fig_acc_c.add_trace(go.Scatter(
                    x=hist_df.get("epoch", range(len(hist_df))),
                    y=hist_df["val_acc"],
                    mode="lines", name="Val Acc",
                    line=dict(color="#AB63FA", dash="dash"),
                ))
                fig_acc_c.update_layout(
                    xaxis_title="Epoch", yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1.05]),
                    margin=dict(l=20, r=20, t=30, b=20), height=380,
                )
                st.plotly_chart(fig_acc_c, use_container_width=True)
            else:
                st.info("Accuracy columns not found in CSV.")

        # Show saved PNG as fallback / cross-check
        img_path = dl.get_curves_path(curve_model, curve_fold)
        if img_path:
            with st.expander("Show saved curves image"):
                st.image(str(img_path), caption=f"curves_fold{curve_fold}.png — {curve_model}")

    else:
        # Try to show the saved PNG if the CSV isn't present
        img_path = dl.get_curves_path(curve_model, curve_fold)
        if img_path:
            st.image(str(img_path), caption=f"curves_fold{curve_fold}.png — {curve_model}", use_column_width=True)
        else:
            st.info(
                f"`history_fold{curve_fold}.csv` not found for **{curve_model}**. "
                f"Download Kaggle outputs and place history CSVs in `results/{dl.EXPERIMENTS[curve_model]['folder']}/`."
            )

    # Side-by-side all-folds comparison
    st.markdown("---")
    st.subheader("All-Fold Val Accuracy Overlay")
    st.markdown("Overlay validation accuracy across all 5 folds for a single experiment to inspect fold variance.")

    overlay_model = st.selectbox("Experiment for overlay", options=dl.MODEL_NAMES, key="cv_overlay")
    all_hists = dl.load_all_histories(overlay_model)

    has_any = any(h is not None for h in all_hists.values())
    if has_any:
        fig_overlay = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        for fold_i, hdf in all_hists.items():
            if hdf is not None and "val_acc" in hdf.columns:
                fig_overlay.add_trace(go.Scatter(
                    x=hdf.get("epoch", range(len(hdf))),
                    y=hdf["val_acc"],
                    mode="lines",
                    name=f"Fold {fold_i}",
                    line=dict(color=colors[fold_i - 1]),
                ))
        if fig_overlay.data:
            fig_overlay.update_layout(
                xaxis_title="Epoch", yaxis_title="Val Accuracy",
                yaxis=dict(range=[0, 1.05]),
                margin=dict(l=20, r=20, t=40, b=20), height=400,
            )
            st.plotly_chart(fig_overlay, use_container_width=True)
        else:
            st.info("No `val_acc` column found in history CSVs.")
    else:
        st.info(f"No history CSVs found for **{overlay_model}**.")

# ─── TAB 3: SUMMARY TABLE ────────────────────────────────────────────────
with tab_summary:
    st.subheader("Cross-Validation Summary — All Experiments")

    tbl = filt_sum.copy()
    tbl["MC_Accuracy"] = tbl.apply(lambda r: f"{r['MC_Accuracy']:.4f} ± {r['Std']:.4f}", axis=1)
    tbl["EER"] = tbl.apply(lambda r: f"{r['EER']:.4f} ± {r['Std_EER']:.4f}", axis=1)
    tbl = tbl.drop(columns=["Std", "Std_EER"])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if metric_choice == "MC Accuracy":
        best = filt_sum.loc[filt_sum["MC_Accuracy"].idxmax()]
        stable = filt_sum.loc[filt_sum["Std"].idxmin()]
        st.success(f"Best mean MC Accuracy: **{best['Model']}** ({best['MC_Accuracy']:.4f})")
        st.info(f"Most stable: **{stable['Model']}** (std = {stable['Std']:.4f})")
    else:
        best = filt_sum.loc[filt_sum["EER"].idxmin()]
        st.success(f"Lowest mean EER: **{best['Model']}** ({best['EER']:.4f})")

# ─── TAB 4: INSIGHTS ─────────────────────────────────────────────────────
with tab_insights:
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### 1. Ranking Is Consistent\n"
            "Across all 5 folds the performance order is preserved: "
            "**Multimodal ≈ Video ≫ Audio Paper ≫ Audio LW**. "
            "Results are not driven by a single favorable split."
        )
    with c2:
        st.markdown(
            "### 2. Video Folds 1–3 Plateau\n"
            "Video Lightweight converges to exactly the same MC accuracy "
            "(0.8923) in Folds 1–3, suggesting the model hits a consistent "
            "performance ceiling on those splits."
        )
    with c3:
        st.markdown(
            "### 3. Multimodal Fold 4 Anomaly\n"
            "A training instability (val_loss spike at epoch 190) in Fold 4 "
            "forced an earlier best-checkpoint save, explaining the slight "
            "accuracy dip relative to other folds."
        )

    st.markdown("---")
    st.markdown(
        "### Interpretation\n"
        "The fold-level analysis confirms that all conclusions are **robust across "
        "cross-validation splits**. The audio model is consistently weak — systematic, "
        "not split-specific. Video and multimodal maintain high performance across all "
        "folds, supporting the reliability of the global ranking."
    )
