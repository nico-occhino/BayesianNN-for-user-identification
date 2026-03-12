"""
overview_dashboard.py — BioVid overall performance summary.
Loads all data from results/*/metrics.json at runtime via data_loader.
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
    page_title="BioVid Dashboard — Overview",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# LOAD DATA
# =========================================================
df = dl.build_summary_df()
fold_df = dl.build_fold_df()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### Overview")
st.sidebar.info(
    "Global comparison across all 4 experimental configurations on the "
    "BioVid speaker identification task (43 identities, 5-fold CV)."
)
dl.sidebar_results_status()

show_table   = st.sidebar.checkbox("Show raw metrics table", value=False)
metric_focus = st.sidebar.selectbox("Highlight metric in table", ["MC Accuracy", "EER"])

# =========================================================
# HEADER
# =========================================================
st.title("BioVid Speaker Identification")
st.markdown("## Overall Performance Summary")
st.markdown(
    "All four experiments — **Audio Lightweight**, **Audio Paper-Exact**, "
    "**Video Lightweight**, **Multimodal Lightweight** — under "
    "**5-fold stratified cross-validation**, 43 speakers, 650 videos."
)

if df.empty:
    st.error("No results found. Copy your `metrics.json` files into the `results/` subfolders.")
    st.stop()

st.divider()

# =========================================================
# KPI CARDS — derived from real data
# =========================================================
best_acc  = df.loc[df["MC_Accuracy"].idxmax()]
best_eer  = df.loc[df["EER"].idxmin()]
video_row = df[df["Model"] == "Video Lightweight"]
audio_row = df[df["Model"] == "Audio Lightweight"]
multi_row = df[df["Model"] == "Multimodal Lightweight"]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Best MC Accuracy",
              f"{best_acc['MC_Accuracy']:.4f} ± {best_acc['Std']:.4f}",
              best_acc["Model"])
with c2:
    st.metric("Lowest EER",
              f"{best_eer['EER']:.4f}",
              best_eer["Model"])
with c3:
    if not video_row.empty and not audio_row.empty:
        gain = video_row.iloc[0]["MC_Accuracy"] - audio_row.iloc[0]["MC_Accuracy"]
        st.metric("Video vs Audio LW Gain", f"+{gain:.4f}", "same 153K params")
    else:
        st.metric("Parameters Best Model", best_acc["Parameters"], best_acc["Model"])
with c4:
    if not multi_row.empty and not video_row.empty:
        eer_red = (video_row.iloc[0]["EER"] - multi_row.iloc[0]["EER"]) / video_row.iloc[0]["EER"]
        st.metric("EER Reduction Video→Multi", f"{eer_red:.1%}", "relative improvement")
    else:
        st.metric("Models Evaluated", str(len(df)), "experiments")

st.divider()

# =========================================================
# TABS
# =========================================================
tab_perf, tab_radar, tab_cv, tab_table, tab_images, tab_insights = st.tabs([
    "📊 Performance", "🕸 Radar", "📈 Cross-Validation", "🗂 Table", "🖼 Images", "💡 Insights"
])

# ─── TAB 1: PERFORMANCE ──────────────────────────────────────────────────
with tab_perf:
    left, right = st.columns(2)

    with left:
        st.subheader("Mean MC Accuracy (± std)")
        fig_acc = px.bar(
            df, x="Model", y="MC_Accuracy",
            color="Modality",
            color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
            text=df["MC_Accuracy"].map(lambda x: f"{x:.4f}"),
            error_y="Std",
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(
            yaxis=dict(range=[0, 1.05]), xaxis_tickangle=-15,
            margin=dict(l=20, r=20, t=40, b=60), height=480,
            legend_title="Modality",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with right:
        st.subheader("Equal Error Rate (lower = better)")
        fig_eer = px.bar(
            df, x="Model", y="EER",
            color="Modality",
            color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
            text=df["EER"].map(lambda x: f"{x:.4f}"),
            error_y="Std_EER",
        )
        fig_eer.update_traces(textposition="outside")
        fig_eer.update_layout(
            xaxis_tickangle=-15,
            margin=dict(l=20, r=20, t=40, b=60), height=480,
        )
        st.plotly_chart(fig_eer, use_container_width=True)

    st.subheader("Accuracy vs EER Trade-off Space")
    fig_scatter = px.scatter(
        df,
        x="EER", y="MC_Accuracy",
        size="MC_Accuracy", color="Modality",
        color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
        hover_name="Model",
        hover_data={"Parameters": True, "Std": True, "Std_EER": True},
        text="Model",
        size_max=60,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        xaxis_title="EER (lower is better)",
        yaxis_title="Mean MC Accuracy (higher is better)",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20), height=480,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Efficiency chart: accuracy vs parameter count
    st.subheader("Efficiency: Accuracy vs Parameter Count")
    param_num = df["Parameters"].map({"153K": 153, "307K": 307, "25M": 25000})
    fig_eff = px.scatter(
        df.assign(Params_K=param_num),
        x="Params_K", y="MC_Accuracy",
        color="Modality",
        color_discrete_map={"Audio": "#EF553B", "Video": "#00CC96", "Multimodal": "#636EFA"},
        size="MC_Accuracy",
        hover_name="Model",
        text="Model",
        log_x=True,
        size_max=50,
    )
    fig_eff.update_traces(textposition="top center")
    fig_eff.update_layout(
        xaxis_title="Parameter Count (K, log scale)",
        yaxis_title="Mean MC Accuracy",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20), height=420,
    )
    st.plotly_chart(fig_eff, use_container_width=True)
    st.caption(
        "Video LW achieves higher accuracy than Audio Paper-Exact using **163× fewer parameters**. "
        "Efficiency outlier = Video Lightweight."
    )

# ─── TAB 2: RADAR ────────────────────────────────────────────────────────
with tab_radar:
    st.subheader("Multi-Metric Radar Comparison")
    st.markdown(
        "Each axis is normalized 0–1 (higher = better on all axes). "
        "EER and Std axes are inverted: `1 − normalized_value`."
    )

    if len(df) >= 2:
        categories = ["MC Accuracy", "Stability (1−Std)", "Verification (1−EER)"]

        max_acc = df["MC_Accuracy"].max()
        max_std = df["Std"].max()
        max_eer = df["EER"].max()

        fig_radar = go.Figure()
        for _, row in df.iterrows():
            vals = [
                row["MC_Accuracy"] / max_acc if max_acc > 0 else 0,
                1 - (row["Std"] / max_std) if max_std > 0 else 1,
                1 - (row["EER"] / max_eer) if max_eer > 0 else 1,
            ]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself", name=row["Model"],
                line_color=dl.MODEL_COLORS.get(row["Model"], "#888"),
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=500,
            margin=dict(l=60, r=60, t=60, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("At least 2 experiments needed for radar chart.")

# ─── TAB 3: CROSS-VALIDATION ─────────────────────────────────────────────
with tab_cv:
    st.subheader("Fold-Level MC Accuracy — All Models")
    if not fold_df.empty:
        selected = st.multiselect(
            "Select models",
            options=fold_df["Model"].unique().tolist(),
            default=fold_df["Model"].unique().tolist(),
        )
        filt = fold_df[fold_df["Model"].isin(selected)]

        fig_line = px.line(
            filt, x="Fold", y="MC_Accuracy", color="Model",
            color_discrete_map=dl.MODEL_COLORS,
            markers=True,
        )
        fig_line.update_layout(
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=40, b=20), height=440,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        fig_eer_line = px.line(
            filt, x="Fold", y="EER", color="Model",
            color_discrete_map=dl.MODEL_COLORS,
            markers=True,
        )
        fig_eer_line.update_layout(
            yaxis_title="EER",
            margin=dict(l=20, r=20, t=40, b=20), height=380,
        )
        st.plotly_chart(fig_eer_line, use_container_width=True)
    else:
        st.info("No fold data available yet.")

# ─── TAB 4: TABLE ────────────────────────────────────────────────────────
with tab_table:
    st.subheader("Full Metrics Table")
    styled = df.copy()
    styled["MC_Accuracy"] = styled.apply(lambda r: f"{r['MC_Accuracy']:.4f} ± {r['Std']:.4f}", axis=1)
    styled["EER"] = styled.apply(lambda r: f"{r['EER']:.4f} ± {r['Std_EER']:.4f}", axis=1)
    styled = styled.drop(columns=["Std", "Std_EER"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if metric_focus == "MC Accuracy":
        best = df.loc[df["MC_Accuracy"].idxmax()]
        st.success(f"🏆 Best MC Accuracy: **{best['Model']}** — {best['MC_Accuracy']:.4f} ± {best['Std']:.4f}")
    else:
        best = df.loc[df["EER"].idxmin()]
        st.success(f"🏆 Best EER: **{best['Model']}** — {best['EER']:.4f} ± {best['Std_EER']:.4f}")

# ─── TAB 5: IMAGES ───────────────────────────────────────────────────────
with tab_images:
    st.subheader("Confusion Matrices")
    available_models = [n for n in dl.MODEL_NAMES if dl.get_confusion_matrix_path(n) is not None]
    if available_models:
        model_sel = st.selectbox("Select experiment", available_models)
        img_path = dl.get_confusion_matrix_path(model_sel)
        if img_path:
            st.image(str(img_path), caption=f"Confusion matrix — {model_sel}", use_column_width=True)
    else:
        st.info(
            "No confusion matrix PNGs found in `results/` yet. "
            "Download Kaggle outputs and place `confusion_matrix_all_folds.png` "
            "in the corresponding results subfolder."
        )

# ─── TAB 6: INSIGHTS ─────────────────────────────────────────────────────
with tab_insights:
    st.subheader("Key Findings")

    if not df.empty:
        multi = df[df["Model"] == "Multimodal Lightweight"]
        video = df[df["Model"] == "Video Lightweight"]
        audio = df[df["Model"] == "Audio Lightweight"]
        paper = df[df["Model"] == "Audio Paper-Exact"]

        c1, c2, c3 = st.columns(3)
        with c1:
            if not multi.empty:
                st.markdown(
                    f"### 🏆 Best Overall Model\n"
                    f"**Multimodal Lightweight** achieves **{multi.iloc[0]['MC_Accuracy']:.4f}** "
                    f"MC accuracy and **{multi.iloc[0]['EER']:.4f}** EER from only "
                    f"~{multi.iloc[0]['Parameters']} parameters."
                )
        with c2:
            if not video.empty and not paper.empty:
                diff = video.iloc[0]["MC_Accuracy"] - paper.iloc[0]["MC_Accuracy"]
                st.markdown(
                    f"### ⚡ Efficiency Paradox\n"
                    f"**Video LW** ({video.iloc[0]['Parameters']}) outperforms "
                    f"**Audio Paper** (25M) by **+{diff:.4f}** accuracy — "
                    f"using **163× fewer parameters**."
                )
        with c3:
            if not video.empty and not audio.empty:
                diff = video.iloc[0]["MC_Accuracy"] - audio.iloc[0]["MC_Accuracy"]
                st.markdown(
                    f"### 👁 Modality Dominance\n"
                    f"**Video** outperforms **Audio** (same architecture, same param count) "
                    f"by **+{diff:.4f}** accuracy. Modality quality dominates capacity."
                )

    st.markdown("---")
    st.markdown(
        """
        ### Interpretation
        Visual lip dynamics are the primary discriminative signal in BioVid. Audio contributes
        complementary information only when fused — it cannot compensate for lack of visual data,
        even when model capacity is dramatically increased. MC Dropout provides its largest gain
        on the Video model, where high-quality features are present but single-pass predictions
        are noisy due to small per-class sample counts.
        """
    )

# =========================================================
# OPTIONAL RAW TABLE
# =========================================================
if show_table:
    st.divider()
    st.subheader("Raw Metrics")
    st.dataframe(df, use_container_width=True, hide_index=True)
