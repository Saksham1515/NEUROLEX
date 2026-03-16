"""Page 10 — Bias & Toxicity Detection"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS
from app import render_sidebar
render_sidebar()    

if "selected_text" not in st.session_state:
    st.session_state["selected_text"] = None
    
st.set_page_config(page_title="Bias & Toxicity | NEUROLEX", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#2EC4B6,#0B7A75)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Hide Streamlit auto multipage navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
styled_header("🛡️", "Bias & Toxicity Detection", "Multi-label toxicity scoring using ToxicBERT — 6-category classification", "#2EC4B6")
no_model_warning(MODELS["toxicity"]["model_name"])

EXAMPLES = [
    ("✅ Safe Text", "The research team published their findings in Nature journal yesterday."),
    ("⚠️ Borderline", "This is the worst implementation I have ever seen in my life."),
]

st.markdown("""
<div style="background:#161B22;border:1px solid #30363D;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
  <p style="color:#8B949E;font-size:0.82em;margin:0;">
    🛡️ <b style="color:#2EC4B6;">ToxicBERT</b> classifies text into 6 toxicity categories from the Jigsaw dataset.
    Scores are probabilities (0–1). Results are for research and content moderation use only.
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("**📝 Text to Analyze**")
    ex_cols = st.columns(2)
    for i, (ec, (label, ex)) in enumerate(zip(ex_cols, EXAMPLES)):
        with ec:
            if st.button(f"{label}", use_container_width=True):
                st.session_state.selected_text = ex

    text = st.text_area(
        "Input", value=st.session_state.selected_text or "", height=160,
        placeholder="Enter text to analyze for toxicity and bias...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**⚙️ Options**")
    mode = st.radio("Mode", ["Single Text", "Multi-Text (one per line)"])
    show_threshold = st.checkbox("Show threshold annotations", value=True)
    run = st.button("🛡️ Analyze Toxicity", use_container_width=True)

    st.markdown("""
    <div style="background:#21262D;border-radius:8px;padding:10px;margin-top:8px;">
      <p style="color:#8B949E;font-size:0.8em;margin:0 0 4px;"><b>Thresholds:</b></p>
      <p style="color:#06D6A0;font-size:0.78em;margin:2px 0;">● Safe: &lt; 15%</p>
      <p style="color:#FFB703;font-size:0.78em;margin:2px 0;">● Warning: 15–40%</p>
      <p style="color:#EF233C;font-size:0.78em;margin:2px 0;">● Flagged: &gt; 40%</p>
    </div>
    """, unsafe_allow_html=True)

def render_toxicity_result(result: dict, text_preview: str = ""):
    from neurolex.modules.toxicity import CATEGORY_INFO, SAFE_THRESHOLD, WARN_THRESHOLD

    verdict = result.get("overall_verdict", "Unknown")
    risk_color = result.get("risk_color", "#8B949E")
    scores = result.get("scores", {})
    cat_info = result.get("category_info", CATEGORY_INFO)

    if text_preview:
        st.markdown(f"**Text:** *{text_preview[:80]}{'…' if len(text_preview)>80 else ''}*")

    st.markdown(f"""
    <div style="background:{risk_color}22;border:2px solid {risk_color};border-radius:12px;
                padding:16px;text-align:center;margin:8px 0;">
      <h3 style="color:{risk_color};margin:0;font-size:1.4em;">{verdict}</h3>
      <p style="color:#8B949E;font-size:0.82em;margin:4px 0 0;">
        Max score: {result.get('max_score', 0):.1%}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Radar / Bar chart
    if scores:
        labels = list(cat_info[c]["label"] for c in scores.keys() if c in cat_info)
        values = list(scores.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=[
                "#EF233C" if v >= WARN_THRESHOLD
                else "#FFB703" if v >= SAFE_THRESHOLD
                else "#06D6A0"
                for v in values
            ],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
        if show_threshold:
            fig.add_hline(y=SAFE_THRESHOLD, line_dash="dash", line_color="#06D6A0",
                          annotation_text="Safe threshold", annotation_font_color="#06D6A0")
            fig.add_hline(y=WARN_THRESHOLD, line_dash="dash", line_color="#EF233C",
                          annotation_text="Flag threshold", annotation_font_color="#EF233C")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3", family="Inter"),
            xaxis=dict(gridcolor="#30363D"),
            yaxis=dict(gridcolor="#30363D", range=[0, 1.1], tickformat=".0%"),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Category cards
    cols = st.columns(3)
    for i, (cat, score) in enumerate(scores.items()):
        info = cat_info.get(cat, {})
        color = info.get("color", "#8B949E")
        icon = info.get("icon", "🔹")
        label = info.get("label", cat)
        desc = info.get("desc", "")
        badge_color = "#EF233C" if score >= WARN_THRESHOLD else "#FFB703" if score >= SAFE_THRESHOLD else "#06D6A0"
        badge_text = "Flagged" if score >= WARN_THRESHOLD else "Warning" if score >= SAFE_THRESHOLD else "Safe"
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
                        padding:12px;margin:4px 0;border-top:2px solid {badge_color};">
              <div style="font-size:1.2em;">{icon}</div>
              <b style="color:#E6EDF3;font-size:0.88em;">{label}</b>
              <p style="color:#8B949E;font-size:0.75em;margin:2px 0;">{desc}</p>
              <div style="display:flex;justify-content:space-between;margin-top:6px;">
                <span style="color:{badge_color};font-weight:700;font-size:0.9em;">{score:.1%}</span>
                <span style="background:{badge_color}22;color:{badge_color};border-radius:8px;
                             padding:1px 8px;font-size:0.75em;">{badge_text}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

if run:
    if not text.strip():
        st.warning("Please enter text to analyze.")
    else:
        from neurolex.modules.toxicity import ToxicityDetector
        detector = ToxicityDetector()

        st.markdown("---")

        if mode == "Multi-Text (one per line)":
            texts = [t.strip() for t in text.strip().split("\n") if t.strip()]
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results = detector.batch_detect(texts)
            for txt, res in zip(texts, results):
                with st.expander(f"🔍 {txt[:60]}…"):
                    render_toxicity_result(res, txt)
        else:
            with st.spinner("Running toxicity classification..."):
                result = detector.detect(text)
            render_toxicity_result(result)

            with st.expander("📋 Detailed Category Analysis"):
                threshold_analysis = detector.safe_threshold_analysis(result.get("scores", {}))
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Category": k,
                        "Score": f"{v['score']:.4f}",
                        "Safe": "✅" if v["safe"] else "",
                        "Warning": "⚠️" if v["warning"] else "",
                        "Flagged": "🚨" if v["flagged"] else "",
                    }
                    for k, v in threshold_analysis.items()
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
