"""
modality_dashboard.py — Audio vs Video vs Multimodal (all lightweight, equal capacity).
Dynamic data from metrics.json + training curve viewer.
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
    page_title="BioVid Dashboard — Modality",
    page_icon="🎛️",
    layout="wide",
)

# =========================================================
# LOAD DATA (lightweight models only)
# =========================================================
MODALITY_MODELS = ["Audio Lightweight", "Video Lightweight", "Multimodal Lightweight"]

summary_df = dl.build_summary_df()
fold_df    = dl.build_fold_df()

mod_summary = summary_df[summary_df["Model"].isin(MODALITY_MODELS)].copy()
mod_folds   = fold_df[fold_df["Model"].isin(MODALITY_MODELS)].copy()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### Modality Comparison")
st.sidebar.info(
    "All three models share the identical **Lightweight GAP architecture (~153K params)**. "
    "Any performance difference is due to modality alone, not capacity."
)
dl.sidebar_results_status()

show_table   = st.sidebar.checkbox("Show raw metrics table", value=False)
metric_focus = st.sidebar.selectbox("Highlight metric", ["MC Accuracy", "EER", "Uncertainty"])

# =========================================================
# HEADER
# =========================================================
st.title("BioVid Speaker Identification")
st.markdown("## Modality Comparison")
st.markdown(
    "**Audio**, **Video**, and **Multimodal** — all using the identical "
    "Lightweight GAP architecture (~153K parameters). "
    "Performance differences reflect modality quality, not model capacity."
)

if mod_summary.empty:
    st.error("No results found. Copy `metrics.json` files into the `results/` subfolders.")
    st.stop()

st.divider()

# =========================================================
# KPI CARDS
# =========================================================
best_acc = mod_summary.loc[mod_summary["MC_Accuracy"].idxmax()]
best_eer = mod_summary.loc[mod_summary["EER"].idxmin()]

audio = mod_summary[mod_summary["Model"] == "Audio Lightweight"]
video = mod_summary[mod_summary["Model"] == "Video Lightweight"]
multi = mod_summary[mod_summary["Model"] == "Multimodal Lightweight"]

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
    if not video.empty and not audio.empty:
        gain = video.iloc[0]["MC_Accuracy"] - audio.iloc[0]["MC_Accuracy"]
        st.metric("Video vs Audio Gain", f"+{gain:.4f}", "same architecture")
    else:
        st.metric("Video vs Audio", "N/A", "")
with c4:
    if not multi.empty and not video.empty:
        gain = multi.iloc[0]["MC_Accuracy"] - video.iloc[0]["MC_Accuracy"]
        st.metric("Multimodal vs Video Gain", f"+{gain:.4f}", "fusion benefit")
    else:
        st.metric("Multimodal vs Video", "N/A", "")

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
        st.subheader("Mean MC Accuracy (± std)")
        fig_acc = px.bar(
            mod_summary, x="Modality", y="MC_Accuracy",
            color="Modality",
            color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
            text=mod_summary["MC_Accuracy"].map(lambda x: f"{x:.4f}"),
            error_y="Std",
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20), height=480,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with right:
        st.subheader("Equal Error Rate (lower = better)")
        fig_eer = px.bar(
            mod_summary, x="Modality", y="EER",
            color="Modality",
            color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
            text=mod_summary["EER"].map(lambda x: f"{x:.4f}"),
            error_y="Std_EER",
        )
        fig_eer.update_traces(textposition="outside")
        fig_eer.update_layout(
            margin=dict(l=20, r=20, t=40, b=20), height=480,
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    st.subheader("Accuracy vs EER Trade-off")
    fig_scatter = px.scatter(
        mod_summary, x="EER", y="MC_Accuracy",
        size="MC_Accuracy", color="Modality",
        color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
        hover_name="Model",
        hover_data={"Std": True, "Std_EER": True},
        text="Modality",
        size_max=60,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20), height=460,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # EER reduction table
    if not audio.empty and not video.empty and not multi.empty:
        st.subheader("EER Reduction Chain")
        audio_eer = audio.iloc[0]["EER"]
        video_eer = video.iloc[0]["EER"]
        multi_eer = multi.iloc[0]["EER"]

        import pandas as pd
        eer_chain = pd.DataFrame({
            "Comparison": ["Audio → Video", "Video → Multimodal", "Audio → Multimodal"],
            "EER From": [f"{audio_eer:.4f}", f"{video_eer:.4f}", f"{audio_eer:.4f}"],
            "EER To":   [f"{video_eer:.4f}", f"{multi_eer:.4f}", f"{multi_eer:.4f}"],
            "Relative Reduction": [
                f"{(audio_eer - video_eer) / audio_eer:.1%}",
                f"{(video_eer - multi_eer) / video_eer:.1%}",
                f"{(audio_eer - multi_eer) / audio_eer:.1%}",
            ],
        })
        st.dataframe(eer_chain, use_container_width=True, hide_index=True)

# ─── TAB 2: CROSS-VALIDATION ─────────────────────────────────────────────
with tab_cv:
    st.subheader("Fold-Level Accuracy")
    if not mod_folds.empty:
        fig_line = px.line(
            mod_folds, x="Fold", y="MC_Accuracy", color="Model",
            color_discrete_map=dl.MODEL_COLORS,
            markers=True,
        )
        fig_line.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20), height=420,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        fig_eer_fold = px.line(
            mod_folds, x="Fold", y="EER", color="Model",
            color_discrete_map=dl.MODEL_COLORS,
            markers=True,
        )
        fig_eer_fold.update_layout(
            yaxis_title="EER",
            margin=dict(l=20, r=20, t=40, b=20), height=360,
        )
        st.plotly_chart(fig_eer_fold, use_container_width=True)
    else:
        st.info("No fold data available yet.")

# ─── TAB 3: TRAINING CURVES ──────────────────────────────────────────────
with tab_curves:
    st.subheader("Training History")
    st.markdown("Inspect loss and accuracy curves for each modality experiment.")

    col_m, col_f = st.columns([2, 1])
    with col_m:
        curve_model = st.selectbox("Experiment", MODALITY_MODELS, key="mod_curve_model")
    with col_f:
        curve_fold = st.selectbox("Fold", [1, 2, 3, 4, 5], key="mod_curve_fold")

    hist = dl.load_history(curve_model, curve_fold)
    if hist is not None:
        cl, cr = st.columns(2)
        with cl:
            if "train_loss" in hist.columns and "val_loss" in hist.columns:
                fig_loss = go.Figure()
                ep = hist.get("epoch", range(len(hist)))
                fig_loss.add_trace(go.Scatter(x=ep, y=hist["train_loss"],
                                              mode="lines", name="Train",
                                              line=dict(color=dl.MODEL_COLORS.get(curve_model, "#636EFA"))))
                fig_loss.add_trace(go.Scatter(x=ep, y=hist["val_loss"],
                                              mode="lines", name="Val",
                                              line=dict(color="#EF553B", dash="dash")))
                fig_loss.update_layout(
                    title=f"Loss — Fold {curve_fold}",
                    xaxis_title="Epoch", yaxis_title="Loss",
                    margin=dict(l=20, r=20, t=50, b=20), height=380,
                )
                st.plotly_chart(fig_loss, use_container_width=True)
        with cr:
            if "train_acc" in hist.columns and "val_acc" in hist.columns:
                fig_a = go.Figure()
                ep = hist.get("epoch", range(len(hist)))
                fig_a.add_trace(go.Scatter(x=ep, y=hist["train_acc"],
                                           mode="lines", name="Train",
                                           line=dict(color=dl.MODEL_COLORS.get(curve_model, "#636EFA"))))
                fig_a.add_trace(go.Scatter(x=ep, y=hist["val_acc"],
                                           mode="lines", name="Val",
                                           line=dict(color="#EF553B", dash="dash")))
                fig_a.update_layout(
                    title=f"Accuracy — Fold {curve_fold}",
                    xaxis_title="Epoch", yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1.05]),
                    margin=dict(l=20, r=20, t=50, b=20), height=380,
                )
                st.plotly_chart(fig_a, use_container_width=True)
    else:
        img = dl.get_curves_path(curve_model, curve_fold)
        if img:
            st.image(str(img), use_column_width=True)
        else:
            st.info(f"`history_fold{curve_fold}.csv` not found for **{curve_model}**.")

# ─── TAB 4: TABLE ────────────────────────────────────────────────────────
with tab_table:
    st.subheader("Modality Metrics")
    tbl = mod_summary.copy()
    tbl["MC_Accuracy"] = tbl.apply(lambda r: f"{r['MC_Accuracy']:.4f} ± {r['Std']:.4f}", axis=1)
    tbl["EER"] = tbl.apply(lambda r: f"{r['EER']:.4f} ± {r['Std_EER']:.4f}", axis=1)
    tbl = tbl.drop(columns=["Std", "Std_EER"])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if not audio.empty and not video.empty and not multi.empty:
        import pandas as pd
        rel = pd.DataFrame({
            "Comparison": ["Video vs Audio", "Multimodal vs Audio", "Multimodal vs Video"],
            "Accuracy Gain": [
                f"+{video.iloc[0]['MC_Accuracy'] - audio.iloc[0]['MC_Accuracy']:.4f}",
                f"+{multi.iloc[0]['MC_Accuracy'] - audio.iloc[0]['MC_Accuracy']:.4f}",
                f"+{multi.iloc[0]['MC_Accuracy'] - video.iloc[0]['MC_Accuracy']:.4f}",
            ],
        })
        st.markdown("### Relative Accuracy Gains")
        st.dataframe(rel, use_container_width=True, hide_index=True)

# ─── TAB 5: INSIGHTS ─────────────────────────────────────────────────────
with tab_insights:
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### 1. Video Dominates Audio\n"
            "Under identical architecture and parameter count, "
            "**video dramatically outperforms audio**. "
            "This is a modality quality result — the visual lip stream "
            "contains richer, lower-noise identity cues in BioVid."
        )
    with c2:
        st.markdown(
            "### 2. Multimodal Is Best\n"
            "Late fusion consistently outperforms video-only, confirming that "
            "audio contributes complementary signal when **combined** — even though "
            "it fails as a standalone modality."
        )
    with c3:
        st.markdown(
            "### 3. Audio Alone Is Insufficient\n"
            "The lightweight audio model essentially operates at near-chance level. "
            "This is not fixable by the same architecture — it requires either "
            "significantly more capacity (see Audio Paper) or a cleaner recording environment."
        )
    st.markdown("---")
    st.markdown(
        "### Interpretation\n"
        "Visual lip dynamics are the primary discriminative signal in BioVid. "
        "Audio contributes complementary information only when fused — it cannot "
        "substitute for visual data at the same capacity level. "
        "Multimodal late fusion is the optimal architecture for this dataset."
    )

if show_table:
    st.divider()
    st.subheader("Raw Metrics")
    st.dataframe(mod_summary, use_container_width=True, hide_index=True)
