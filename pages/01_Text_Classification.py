"""Page 01 — Multi-Label Text Classification"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, progress_bar_html, no_model_warning
from neurolex.config import MODELS

st.set_page_config(page_title="Text Classification | NEUROLEX", page_icon="🏷️", layout="wide")

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#6C63FF,#4ECDC4)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
.stSelectbox>div>div{background:#21262D!important;border:1px solid #30363D!important;color:#E6EDF3!important;}
[data-testid="stMetricValue"]{color:#6C63FF!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("🏷️", "Text Classification", "Multi-label zero-shot classification using BART-MNLI NLI", "#6C63FF")

no_model_warning(MODELS["classifier"]["model_name"])

# ─── Examples ────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Apple has released a new MacBook Pro with M4 chip. The stock surged 3% on the news.",
    "Scientists have discovered a new exoplanet in the habitable zone, potentially supporting liquid water.",
    "The government announced new climate policies to reduce carbon emissions by 40% by 2030.",
]
DEFAULT_LABELS = "Technology, Science, Politics, Business, Environment, Health, Sports, Entertainment"

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("**📄 Input Text**")
    eg_cols = st.columns(3)
    selected_text = None
    for i, (ec, ex) in enumerate(zip(eg_cols, EXAMPLES)):
        with ec:
            if st.button(f"Example {i+1}", key=f"clf_ex_{i}", use_container_width=True):
                selected_text = ex

    text = st.text_area(
        "Enter text to classify",
        value=selected_text or "",
        height=160,
        placeholder="Paste any text here...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**🏷️ Candidate Labels** (comma-separated)")
    labels_str = st.text_area(
        "Labels",
        value=DEFAULT_LABELS,
        height=80,
        label_visibility="collapsed",
    )
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    multi_label = st.checkbox("Multi-label mode", value=True)
    run = st.button("🚀 Classify", use_container_width=True)

# ─── Results ─────────────────────────────────────────────────────────────────
if run and text.strip():
    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        st.error("Please enter at least one label.")
    else:
        with st.spinner("Running zero-shot classification..."):
            from neurolex.modules.classifier import MultiLabelClassifier
            clf = MultiLabelClassifier()
            result = clf.classify(text, labels, threshold=threshold, multi_label=multi_label)

        if result:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("🥇 Top Label", result["top_label"])
            c2.metric("📊 Top Score", f"{result['top_score']:.1%}")
            c3.metric("✅ Above Threshold", len(result["above_threshold"]))

            tab1, tab2, tab3 = st.tabs(["📊 All Scores", "✅ Active Labels", "🔬 Explanation"])
            with tab1:
                scores = result["results"]
                sorted_items = sorted(scores.items(), key=lambda x: -x[1])
                html = ""
                for label, score in sorted_items:
                    color = "#6C63FF" if score >= threshold else "#8B949E"
                    html += progress_bar_html(score, color=color, label=label)
                st.markdown(html, unsafe_allow_html=True)

                fig = go.Figure(go.Bar(
                    x=[s for _, s in sorted_items],
                    y=[l for l, _ in sorted_items],
                    orientation="h",
                    marker=dict(
                        color=[s for _, s in sorted_items],
                        colorscale=[[0, "#21262D"], [1, "#6C63FF"]],
                        showscale=False,
                    ),
                    text=[f"{s:.1%}" for _, s in sorted_items],
                    textposition="outside",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
                    font=dict(color="#E6EDF3", family="Inter"),
                    yaxis=dict(autorange="reversed", gridcolor="#30363D"),
                    xaxis=dict(gridcolor="#30363D", tickformat=".0%"),
                    margin=dict(l=10, r=60, t=10, b=10), height=max(200, len(scores) * 40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if result["above_threshold"]:
                    for label, score in sorted(result["above_threshold"].items(), key=lambda x: -x[1]):
                        st.markdown(
                            f'<div style="background:#6C63FF22;border:1px solid #6C63FF44;'
                            f'border-radius:8px;padding:10px 14px;margin:4px 0;">'
                            f'<b style="color:#6C63FF;">{label}</b> — '
                            f'<span style="color:#E6EDF3;">{score:.2%} confidence</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No labels exceeded the threshold. Try lowering the threshold.")

            with tab3:
                with st.spinner("Computing sentence-level attributions..."):
                    top_labels = list(result["results"].keys())[:3]
                    attributions = clf.explain(text, top_labels)
                uncertainty = clf.uncertainty_score(result["results"])
                st.metric("🎯 Uncertainty Score", f"{uncertainty:.3f}",
                          help="High uncertainty = good candidate for active labeling")
                for label, sents in attributions.items():
                    if sents:
                        st.markdown(f"**{label}** — most influential sentence:")
                        st.markdown(
                            f'<div style="background:#21262D;border-radius:6px;padding:8px 12px;'
                            f'font-size:0.9em;color:#E6EDF3;border-left:3px solid #6C63FF;">'
                            f'{sents[0][0]} <span style="color:#6C63FF;">({sents[0][1]:.2%})</span></div>',
                            unsafe_allow_html=True,
                        )

elif run:
    st.warning("Please enter some text to classify.")
