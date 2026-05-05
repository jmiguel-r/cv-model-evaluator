"""
streamlit_app.py
----------------
Interactive dashboard for CV model evaluation results.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Model Evaluator",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 CV Model Evaluation Dashboard")
st.caption("Systematic analysis of computer vision model predictions — failure modes, metrics & annotation QA")

# ── Sidebar: load results ─────────────────────────────────────────────────────
st.sidebar.header("Load Results")

outputs_dir = Path("outputs")
eval_files  = sorted(outputs_dir.glob("**/*_eval.json")) if outputs_dir.exists() else []
qa_files    = sorted(outputs_dir.glob("**/qa_report.json")) if outputs_dir.exists() else []

if not eval_files:
    st.info(
        "No evaluation results found.\n\n"
        "Run `python scripts/run_evaluation.py` first to generate results."
    )
    st.stop()

selected_eval = st.sidebar.selectbox(
    "Evaluation result",
    options=[str(f) for f in eval_files],
    format_func=lambda p: Path(p).stem,
)

# ── Load eval JSON ────────────────────────────────────────────────────────────
with open(selected_eval) as f:
    eval_data = json.load(f)

class_names = eval_data.get("class_names", [])
per_class   = eval_data.get("per_class_ap", {})
conf_matrix = np.array(eval_data.get("confusion_matrix", [])) if eval_data.get("confusion_matrix") else None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_metrics, tab_classes, tab_confusion, tab_failures, tab_qa = st.tabs([
    "📊 Metrics", "🏷️ Per-class AP", "🔀 Confusion Matrix", "🐛 Failure Modes", "✅ Annotation QA"
])

# ── TAB 1: Overall metrics ────────────────────────────────────────────────────
with tab_metrics:
    st.subheader("Overall Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("mAP@50",     f"{eval_data.get('map50', 0):.3f}")
    col2.metric("mAP@50:95",  f"{eval_data.get('map50_95', 0):.3f}")
    col3.metric("Precision",  f"{eval_data.get('precision', 0):.3f}")
    col4.metric("Recall",     f"{eval_data.get('recall', 0):.3f}")
    col5.metric("F1",         f"{eval_data.get('f1', 0):.3f}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total images",       eval_data.get("total_images", "-"))
    c2.metric("GT boxes",           eval_data.get("total_gt_boxes", "-"))
    c3.metric("Predicted boxes",    eval_data.get("total_pred_boxes", "-"))

# ── TAB 2: Per-class AP bar chart ─────────────────────────────────────────────
with tab_classes:
    st.subheader("Per-class Average Precision @ IoU=0.5")

    if per_class:
        df = pd.DataFrame(
            {"Class": list(per_class.keys()), "AP@50": list(per_class.values())}
        ).sort_values("AP@50", ascending=False)

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
        colors = ["#1B6CA8" if v >= 0.5 else "#C04828" for v in df["AP@50"]]
        ax.barh(df["Class"], df["AP@50"], color=colors)
        ax.axvline(x=eval_data.get("map50", 0), color="gray", linestyle="--", linewidth=1, label="mAP@50")
        ax.set_xlabel("AP@50")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.legend()
        ax.set_facecolor("#f9f9f9")
        fig.tight_layout()
        st.pyplot(fig)

        st.dataframe(df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No per-class data available.")

# ── TAB 3: Confusion matrix ───────────────────────────────────────────────────
with tab_confusion:
    st.subheader("Confusion Matrix")

    if conf_matrix is not None and len(conf_matrix) > 0:
        labels = class_names + ["background/FP"] if class_names else [str(i) for i in range(len(conf_matrix))]
        labels = labels[: len(conf_matrix)]

        # Normalize rows
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        norm_cm  = conf_matrix / np.where(row_sums == 0, 1, row_sums)

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
        im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = norm_cm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.5 else "black")

        fig.tight_layout()
        st.pyplot(fig)

        with st.expander("Raw counts"):
            raw_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
            st.dataframe(raw_df)
    else:
        st.info("No confusion matrix data available.")

# ── TAB 4: Failure modes ──────────────────────────────────────────────────────
with tab_failures:
    st.subheader("Failure Mode Analysis")

    failure_files = sorted(outputs_dir.glob("**/failure_report.json")) if outputs_dir.exists() else []

    if not failure_files:
        st.info("No failure report found. Run `python scripts/run_evaluation.py` to generate one.")
    else:
        selected_fr = st.selectbox(
            "Failure report", [str(f) for f in failure_files],
            format_func=lambda p: Path(p).stem,
        )
        with open(selected_fr) as f:
            fr_data = json.load(f)

        summary = fr_data.get("summary", {})
        records = fr_data.get("records", [])

        if summary:
            # Donut chart
            fig, ax = plt.subplots(figsize=(5, 5))
            labels_f = list(summary.keys())
            sizes    = list(summary.values())
            colors_f = ["#C04828", "#1B6CA8", "#E8A020", "#3A9A72", "#8B5EA8", "#888"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels_f, autopct="%1.1f%%",
                colors=colors_f[: len(labels_f)], startangle=140,
                pctdistance=0.82, wedgeprops=dict(width=0.55),
            )
            for at in autotexts:
                at.set_fontsize(9)
            ax.set_title("Failure distribution", fontsize=11)
            fig.tight_layout()

            col_chart, col_table = st.columns([1, 1])
            with col_chart:
                st.pyplot(fig)
            with col_table:
                st.dataframe(
                    pd.DataFrame({"Type": labels_f, "Count": sizes})
                      .sort_values("Count", ascending=False)
                      .reset_index(drop=True),
                    use_container_width=True,
                )

        if records:
            st.divider()
            st.subheader("Browse individual failures")

            failure_types = ["All"] + sorted(set(r["failure_type"] for r in records))
            selected_type = st.selectbox("Filter by type", failure_types)

            filtered = records if selected_type == "All" else [
                r for r in records if r["failure_type"] == selected_type
            ]

            st.caption(f"Showing {len(filtered)} of {len(records)} records")
            df_rec = pd.DataFrame(filtered)
            if "iou" in df_rec.columns:
                df_rec["iou"] = df_rec["iou"].apply(lambda x: f"{x:.3f}" if x is not None else "-")
            st.dataframe(df_rec, use_container_width=True)

# ── TAB 5: Annotation QA ─────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Annotation Quality Report")

    if not qa_files:
        st.info("No QA report found. Run `python scripts/run_annotation_qa.py` to generate one.")
    else:
        with open(qa_files[0]) as f:
            qa_data = json.load(f)

        qa_col1, qa_col2, qa_col3 = st.columns(3)
        qa_col1.metric("Total images",      qa_data.get("total_images", "-"))
        qa_col2.metric("Total annotations", qa_data.get("total_annotations", "-"))
        issue_rate = qa_data.get("issue_rate", 0)
        qa_col3.metric("Issue rate", f"{issue_rate*100:.1f}%",
                        delta=None if issue_rate < 0.05 else "⚠️ Review recommended",
                        delta_color="inverse")

        # Class distribution chart
        class_dist = qa_data.get("class_distribution", {})
        if class_dist:
            st.subheader("Class distribution")
            df_cls = pd.DataFrame(
                {"Class": list(class_dist.keys()), "Count": list(class_dist.values())}
            ).sort_values("Count", ascending=False)

            fig2, ax2 = plt.subplots(figsize=(10, max(3, len(df_cls) * 0.35)))
            ax2.barh(df_cls["Class"], df_cls["Count"], color="#1B6CA8")
            ax2.set_xlabel("Annotation count")
            ax2.invert_yaxis()
            ax2.set_facecolor("#f9f9f9")
            fig2.tight_layout()
            st.pyplot(fig2)

        # Issue breakdown
        issue_summ = qa_data.get("issue_summary", {})
        if issue_summ:
            st.subheader("Issue breakdown")
            st.dataframe(
                pd.DataFrame({
                    "Issue Type": list(issue_summ.keys()),
                    "Count":      list(issue_summ.values()),
                }).sort_values("Count", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
