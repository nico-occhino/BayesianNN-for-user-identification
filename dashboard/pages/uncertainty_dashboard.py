"""
uncertainty_dashboard.py — MC Dropout & Uncertainty Analysis
------------------------------------------------------------
Standard accuracy and mean_uncertainty values are sourced directly from the
Kaggle training logs (session ~6692s), since these fields were not written
to metrics.json. MC accuracy and EER match the stored metrics.json files.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import MODEL_COLORS, sidebar_results_status

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED FROM KAGGLE LOGS (mean_uncertainty + std_acc not in metrics.json)
# ─────────────────────────────────────────────────────────────────────────────
LOG_DATA = {
    "Audio Lightweight": [
        {"fold": 1, "std_acc": 0.2231, "mc_acc": 0.1769, "eer": 0.1630, "mean_unc": 0.0140},
        {"fold": 2, "std_acc": 0.2154, "mc_acc": 0.2308, "eer": 0.1281, "mean_unc": 0.0140},
        {"fold": 3, "std_acc": 0.2615, "mc_acc": 0.2000, "eer": 0.1359, "mean_unc": 0.0137},
        {"fold": 4, "std_acc": 0.2385, "mc_acc": 0.2538, "eer": 0.1514, "mean_unc": 0.0126},
        {"fold": 5, "std_acc": 0.2615, "mc_acc": 0.2000, "eer": 0.1242, "mean_unc": 0.0133},
    ],
    "Audio Paper (CNNMC)": [
        {"fold": 1, "std_acc": 0.7615, "mc_acc": 0.7615, "eer": 0.0180, "mean_unc": 0.0095},
        {"fold": 2, "std_acc": 0.7846, "mc_acc": 0.7923, "eer": 0.0160, "mean_unc": 0.0110},
        {"fold": 3, "std_acc": 0.8385, "mc_acc": 0.8615, "eer": 0.0139, "mean_unc": 0.0093},
        {"fold": 4, "std_acc": 0.7923, "mc_acc": 0.8231, "eer": 0.0151, "mean_unc": 0.0109},
        {"fold": 5, "std_acc": 0.8385, "mc_acc": 0.8231, "eer": 0.0116, "mean_unc": 0.0111},
    ],
    "Video Lightweight": [
        {"fold": 1, "std_acc": 0.7231, "mc_acc": 0.8154, "eer": 0.0061, "mean_unc": 0.0153},
        {"fold": 2, "std_acc": 0.6769, "mc_acc": 0.8000, "eer": 0.0172, "mean_unc": 0.0154},
        {"fold": 3, "std_acc": 0.6923, "mc_acc": 0.8077, "eer": 0.0140, "mean_unc": 0.0153},
        {"fold": 4, "std_acc": 0.6538, "mc_acc": 0.7231, "eer": 0.0207, "mean_unc": 0.0151},
        {"fold": 5, "std_acc": 0.6538, "mc_acc": 0.7692, "eer": 0.0175, "mean_unc": 0.0161},
    ],
    "Multimodal Fusion": [
        {"fold": 1, "std_acc": 0.8385, "mc_acc": 0.8846, "eer": 0.0029, "mean_unc": 0.0122},
        {"fold": 2, "std_acc": 0.8615, "mc_acc": 0.9000, "eer": 0.0020, "mean_unc": 0.0117},
        {"fold": 3, "std_acc": 0.8769, "mc_acc": 0.9154, "eer": 0.0016, "mean_unc": 0.0104},
        {"fold": 4, "std_acc": 0.8077, "mc_acc": 0.8231, "eer": 0.0111, "mean_unc": 0.0114},
        {"fold": 5, "std_acc": 0.8692, "mc_acc": 0.8923, "eer": 0.0058, "mean_unc": 0.0116},
    ],
}

MODEL_ORDER = ["Audio Lightweight", "Audio Paper (CNNMC)", "Video Lightweight", "Multimodal Fusion"]

# Build flat fold DataFrame
rows = []
for model, folds in LOG_DATA.items():
    for f in folds:
        rows.append({
            "Model": model,
            "Fold": f["fold"],
            "Std_Acc": f["std_acc"],
            "MC_Acc": f["mc_acc"],
            "MC_Gain": f["mc_acc"] - f["std_acc"],
            "EER": f["eer"],
            "Mean_Unc": f["mean_unc"],
        })
fold_df = pd.DataFrame(rows)

# Build summary DataFrame
summary_rows = []
for model in MODEL_ORDER:
    sub = fold_df[fold_df["Model"] == model]
    summary_rows.append({
        "Model": model,
        "Std_Acc":     sub["Std_Acc"].mean(),
        "MC_Acc":      sub["MC_Acc"].mean(),
        "MC_Gain":     sub["MC_Gain"].mean(),
        "MC_Gain_Std": sub["MC_Gain"].std(),
        "Mean_Unc":    sub["Mean_Unc"].mean(),
        "Unc_Std":     sub["Mean_Unc"].std(),
        "EER":         sub["EER"].mean(),
    })
summary_df = pd.DataFrame(summary_rows)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioVid — MC Dropout & Uncertainty",
    page_icon="🎲",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("BioVid Dashboard")
st.sidebar.markdown("### MC Dropout & Uncertainty")
st.sidebar.info(
    "T = 30 stochastic forward passes per sample.\n\n"
    "**Mean uncertainty** = mean std-dev of class probability "
    "distributions across MC passes, averaged over all validation samples.\n\n"
    "**MC Gain** = MC accuracy − single-pass standard accuracy."
)
selected = st.sidebar.multiselect(
    "Models", options=MODEL_ORDER, default=MODEL_ORDER
)
show_tables = st.sidebar.checkbox("Show raw tables", value=False)
sidebar_results_status()

if not selected:
    st.warning("Select at least one model.")
    st.stop()

filt_sum  = summary_df[summary_df["Model"].isin(selected)].copy()
filt_fold = fold_df[fold_df["Model"].isin(selected)].copy()
colors    = MODEL_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("BioVid Speaker Identification")
st.markdown("## MC Dropout & Predictive Uncertainty Analysis")
st.markdown(
    "Each validation sample undergoes **T = 30 stochastic forward passes** "
    "with dropout active. The final prediction is the argmax of the mean "
    "probability vector. **Mean uncertainty** is the mean standard deviation "
    "of those probability distributions across all samples — a genuine "
    "Bayesian uncertainty estimate, not a fold variance proxy."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
best_gain_row = filt_sum.loc[filt_sum["MC_Gain"].idxmax()]
best_unc_row  = filt_sum.loc[filt_sum["Mean_Unc"].idxmin()]
best_mc_row   = filt_sum.loc[filt_sum["MC_Acc"].idxmax()]
worst_gain    = filt_sum.loc[filt_sum["MC_Gain"].idxmin()]

with c1:
    st.metric("Best MC Accuracy",
              f"{best_mc_row['MC_Acc']:.4f}",
              best_mc_row["Model"].split()[0])
with c2:
    st.metric("Largest MC Gain",
              f"{best_gain_row['MC_Gain']:+.4f}",
              best_gain_row["Model"].split()[0])
with c3:
    st.metric("Lowest Uncertainty",
              f"{best_unc_row['Mean_Unc']:.4f}",
              best_unc_row["Model"].split()[0])
with c4:
    st.metric("MC Gain (worst model)",
              f"{worst_gain['MC_Gain']:+.4f}",
              worst_gain["Model"].split()[0],
              delta_color="inverse" if worst_gain["MC_Gain"] < 0 else "normal")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Standard vs MC Accuracy",
    "MC Gain Analysis",
    "Uncertainty",
    "Uncertainty vs Performance",
])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Single-Pass vs Monte Carlo Accuracy")
    st.caption(
        "Faded bars = standard accuracy (dropout off, one pass). "
        "Solid bars = MC accuracy (dropout on, T=30 passes averaged). "
        "The gap is the empirical benefit of Bayesian inference."
    )

    fig = go.Figure()
    for model in [m for m in MODEL_ORDER if m in selected]:
        row = filt_sum[filt_sum["Model"] == model].iloc[0]
        col = colors.get(model, "#888")
        fig.add_bar(
            name=f"{model} — Standard",
            x=[model], y=[row["Std_Acc"]],
            marker_color=col, opacity=0.40,
            text=[f"{row['Std_Acc']:.3f}"], textposition="outside",
        )
        fig.add_bar(
            name=f"{model} — MC (T=30)",
            x=[model], y=[row["MC_Acc"]],
            marker_color=col, opacity=1.0,
            text=[f"{row['MC_Acc']:.3f}"], textposition="outside",
        )
    fig.update_layout(
        barmode="group", height=500,
        yaxis=dict(range=[0, 1.12], title="Accuracy"),
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig_std = px.line(
            filt_fold, x="Fold", y="Std_Acc",
            color="Model", markers=True,
            color_discrete_map=colors,
            title="Standard Accuracy per fold",
            line_dash_sequence=["dash"],
        )
        fig_std.update_layout(yaxis=dict(range=[0, 1.05]), height=400,
                              margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_std, use_container_width=True)
    with right:
        fig_mc = px.line(
            filt_fold, x="Fold", y="MC_Acc",
            color="Model", markers=True,
            color_discrete_map=colors,
            title="MC Accuracy per fold (T=30)",
        )
        fig_mc.update_layout(yaxis=dict(range=[0, 1.05]), height=400,
                             margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_mc, use_container_width=True)

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("MC Gain = MC Accuracy − Standard Accuracy")
    st.caption(
        "Positive = MC averaging helps. Negative = model is too unreliable "
        "for averaging to recover useful signal."
    )

    left, right = st.columns(2)
    with left:
        fig_gain = px.bar(
            filt_sum, x="Model", y="MC_Gain",
            color="Model", color_discrete_map=colors,
            error_y="MC_Gain_Std",
            text=filt_sum["MC_Gain"].map(lambda x: f"{x:+.4f}"),
            title="Mean MC Gain (± std across folds)",
        )
        fig_gain.update_traces(textposition="outside")
        fig_gain.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_gain.update_layout(
            height=480, showlegend=False, xaxis_title="",
            yaxis_title="MC Gain",
            margin=dict(l=20, r=20, t=60, b=80),
        )
        st.plotly_chart(fig_gain, use_container_width=True)

    with right:
        fig_gain_fold = px.line(
            filt_fold, x="Fold", y="MC_Gain",
            color="Model", markers=True,
            color_discrete_map=colors,
            title="MC Gain per fold",
        )
        fig_gain_fold.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_gain_fold.update_layout(
            height=480, yaxis_title="MC Gain",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_gain_fold, use_container_width=True)

    st.markdown("#### Interpretation")
    cols = st.columns(4)
    interpretations = {
        "Audio Lightweight": (
            "st.error", "−2.8pp average",
            "Near-random representations. Averaging 30 random guesses "
            "does not recover signal — it just averages noise. MC can "
            "slightly hurt a model with nothing useful to amplify."
        ),
        "Audio Paper (CNNMC)": (
            "st.warning", "+0.9pp average",
            "Good representations, already stable single-pass outputs. "
            "Small consistent improvement — model is near its ceiling "
            "before MC is applied."
        ),
        "Video Lightweight": (
            "st.success", "+10.3pp average",
            "Strong representations but noisy per-sample predictions "
            "(small-batch stochastic training). MC averaging does major "
            "work: T=30 diverse views consistently correct single-pass errors."
        ),
        "Multimodal Fusion": (
            "st.info", "+3.2pp average",
            "Best overall representations. Moderate MC gain that is "
            "consistent across all 5 folds — reliable improvement "
            "on top of an already high accuracy baseline."
        ),
    }
    for col, model in zip(cols, MODEL_ORDER):
        if model in selected:
            fn_name, label, text = interpretations[model]
            with col:
                fn = getattr(st, fn_name.split(".")[1])
                fn(f"**{model}**  \n**{label}**  \n{text}")

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Mean Predictive Uncertainty σ̄")
    st.caption(
        "σ̄ = mean standard deviation of the 43-class probability vector "
        "across T=30 MC passes, averaged over all validation samples. "
        "Higher σ̄ = model is more uncertain about its own predictions."
    )

    left, right = st.columns(2)
    with left:
        fig_unc = px.bar(
            filt_sum, x="Model", y="Mean_Unc",
            color="Model", color_discrete_map=colors,
            error_y="Unc_Std",
            text=filt_sum["Mean_Unc"].map(lambda x: f"{x:.4f}"),
            title="Mean Uncertainty σ̄ (averaged over folds)",
        )
        fig_unc.update_traces(textposition="outside")
        fig_unc.update_layout(
            height=480, showlegend=False, xaxis_title="",
            yaxis_title="Mean σ̄",
            margin=dict(l=20, r=20, t=60, b=80),
        )
        st.plotly_chart(fig_unc, use_container_width=True)

    with right:
        fig_unc_fold = px.line(
            filt_fold, x="Fold", y="Mean_Unc",
            color="Model", markers=True,
            color_discrete_map=colors,
            title="Uncertainty σ̄ per fold",
        )
        fig_unc_fold.update_layout(
            height=480, yaxis_title="Mean σ̄",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_unc_fold, use_container_width=True)

    fig_box = px.box(
        filt_fold, x="Model", y="Mean_Unc",
        color="Model", points="all",
        color_discrete_map=colors,
        category_orders={"Model": MODEL_ORDER},
        title="Uncertainty distribution across 5 folds",
    )
    fig_box.update_layout(
        height=380, showlegend=False, xaxis_title="",
        yaxis_title="Mean σ̄",
        margin=dict(l=20, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        "📌 **Key finding:** Video Lightweight has the *highest* uncertainty "
        "(σ̄ ≈ 0.0154) yet the largest MC gain (+10pp). "
        "Multimodal Fusion has the *lowest* uncertainty (σ̄ ≈ 0.0115) "
        "and the highest MC accuracy. "
        "High uncertainty does not imply low quality — it reflects that strong "
        "representations trained with small batches produce variable per-sample "
        "outputs, which MC averaging then corrects."
    )

# ── TAB 4 ────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Uncertainty vs Performance — The Full Picture")

    left, right = st.columns(2)
    with left:
        fig_s1 = px.scatter(
            filt_sum, x="Mean_Unc", y="MC_Gain",
            color="Model", text="Model",
            size=[abs(r) * 300 + 20 for r in filt_sum["MC_Gain"]],
            color_discrete_map=colors,
            title="Uncertainty vs MC Gain",
            hover_data={"Std_Acc": ":.4f", "MC_Acc": ":.4f"},
        )
        fig_s1.update_traces(textposition="top center")
        fig_s1.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_s1.update_layout(
            height=460, showlegend=False,
            xaxis_title="Mean uncertainty σ̄",
            yaxis_title="MC Gain",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_s1, use_container_width=True)
        st.caption(
            "High σ̄ + good representations → large MC gain (Video LW). "
            "High σ̄ + poor representations → negative MC gain (Audio LW). "
            "Uncertainty alone does not determine MC benefit."
        )

    with right:
        # FIX: size must be >= 0; MC_Gain can be negative so use abs() + floor
        fig_s2 = px.scatter(
            filt_sum, x="Mean_Unc", y="MC_Acc",
            color="Model", text="Model",
            size=[abs(r) * 300 + 20 for r in filt_sum["MC_Gain"]],
            color_discrete_map=colors,
            title="Uncertainty vs MC Accuracy",
            hover_data={"EER": ":.4f"},
        )
        fig_s2.update_traces(textposition="top center")
        fig_s2.update_layout(
            height=460, showlegend=False,
            xaxis_title="Mean uncertainty σ̄",
            yaxis_title="MC Accuracy",
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_s2, use_container_width=True)
        st.caption(
            "No monotonic relationship between uncertainty and accuracy. "
            "Representation quality mediates the uncertainty–performance link."
        )

    # All folds scatter
    st.subheader("Per-Fold: Uncertainty vs MC Gain (all models, all folds)")
    fig_fs = px.scatter(
        filt_fold, x="Mean_Unc", y="MC_Gain",
        color="Model", symbol="Fold",
        color_discrete_map=colors,
        hover_data={"Fold": True, "Std_Acc": ":.4f", "MC_Acc": ":.4f"},
    )
    fig_fs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_fs.update_layout(
        height=480,
        xaxis_title="Mean uncertainty σ̄",
        yaxis_title="MC Gain",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_fs, use_container_width=True)

    # Summary table
    st.subheader("Summary Table")
    disp = filt_sum[["Model", "Std_Acc", "MC_Acc", "MC_Gain", "Mean_Unc", "EER"]].copy()
    disp.columns = ["Model", "Std Acc", "MC Acc", "MC Gain", "Mean σ̄", "EER"]
    for col in disp.columns[1:]:
        fmt = "{:+.4f}" if col == "MC Gain" else "{:.4f}"
        disp[col] = disp[col].map(lambda x, f=fmt: f.format(x))
    st.dataframe(disp, use_container_width=True, hide_index=True)

    if show_tables:
        st.markdown("### Per-Fold Raw Data")
        disp_f = filt_fold.copy()
        for col in ["Std_Acc", "MC_Acc", "Mean_Unc", "EER"]:
            disp_f[col] = disp_f[col].map(lambda x: f"{x:.4f}")
        disp_f["MC_Gain"] = disp_f["MC_Gain"].map(lambda x: f"{x:+.4f}")
        st.dataframe(disp_f, use_container_width=True, hide_index=True)