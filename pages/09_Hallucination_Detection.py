"""Page 09 — Hallucination Detection & Factuality Scoring"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS

st.set_page_config(page_title="Hallucination Detection | NEUROLEX", page_icon="⚠️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#EF233C,#8B1A1A)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("⚠️", "Hallucination Detection", "NLI-based factuality scoring — detects when generated text contradicts the source", "#EF233C")
no_model_warning(MODELS["hallucination"]["model_name"])

EXAMPLES = [
    {
        "source": "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
        "generated": "The Eiffel Tower is in Paris and was constructed in 1889. Its height is 330 meters.",
        "label": "✅ Factual Example",
    },
    {
        "source": "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
        "generated": "The Eiffel Tower is located in London and was built in 1920.",
        "label": "🚨 Hallucination Example",
    },
    {
        "source": "Machine learning models are trained on large datasets to recognize patterns and make predictions.",
        "generated": "Quantum computers are widely used in hospitals to perform surgeries automatically.",
        "label": "🚨 Severe Hallucination",
    },
]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📄 Source / Reference Document**")
    for ex in EXAMPLES[:2]:
        if st.button(f"Load: {ex['label']}", key=f"h_ex_{ex['label'][:5]}", use_container_width=True):
            st.session_state["h_source"] = ex["source"]
            st.session_state["h_generated"] = ex["generated"]

    source = st.text_area(
        "Source", value=st.session_state.get("h_source", ""),
        height=150, placeholder="Paste the reference document / ground truth...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**🤖 Generated / Claim Text**")
    if st.button("Load Severe Hallucination Example", key="h_ex_severe", use_container_width=True):
        st.session_state["h_source"] = EXAMPLES[2]["source"]
        st.session_state["h_generated"] = EXAMPLES[2]["generated"]

    generated = st.text_area(
        "Generated", value=st.session_state.get("h_generated", ""),
        height=150, placeholder="Paste the generated text or claim to verify...",
        label_visibility="collapsed",
    )

mode = st.radio("Analysis Mode", ["📄 Document-Level", "📝 Sentence-Level"], horizontal=True)
run = st.button("⚠️ Analyze Factuality", use_container_width=False)

if run:
    if not source.strip() or not generated.strip():
        st.warning("Please provide both source and generated text.")
    else:
        from neurolex.modules.hallucination import HallucinationDetector
        detector = HallucinationDetector()

        with st.spinner("Running NLI-based factuality analysis..."):
            result = detector.score_factuality(source, generated)

        st.markdown("---")

        # Verdict banner
        st.markdown(f"""
        <div style="background:{result['verdict_color']}22;border:2px solid {result['verdict_color']};
                    border-radius:14px;padding:20px;text-align:center;margin-bottom:16px;">
          <h2 style="color:{result['verdict_color']};font-size:1.8em;margin:0;">{result['verdict']}</h2>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Factuality Score", f"{result['factuality_score']:.1%}",
                  help="P(entailment) — higher is more factual")
        c2.metric("🚨 Hallucination Risk", f"{result['hallucination_risk']:.1%}",
                  help="P(contradiction) — higher means more hallucinated")
        c3.metric("⚠️ Uncertainty", f"{result['uncertainty']:.1%}",
                  help="P(neutral) — insufficient evidence")

        # Gauge chart
        scores_dict = result["scores"]
        fig = go.Figure(go.Bar(
            x=list(scores_dict.values()),
            y=list(scores_dict.keys()),
            orientation="h",
            marker_color=["#06D6A0", "#FFB703", "#EF233C"],
            text=[f"{v:.1%}" for v in scores_dict.values()],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3", family="Inter"),
            xaxis=dict(tickformat=".0%", gridcolor="#30363D", range=[0, 1.1]),
            yaxis=dict(gridcolor="#30363D"),
            margin=dict(l=10, r=70, t=20, b=10), height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

        if "Sentence" in mode:
            with st.spinner("Analyzing sentence-by-sentence..."):
                sent_results = detector.sentence_level_analysis(source, generated)
                agg = detector.aggregate_scores(sent_results)

            if agg:
                st.markdown("**📊 Aggregated Sentence-Level Scores**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Factuality", f"{agg.get('mean_factuality', 0):.1%}")
                c2.metric("Avg Hallucination Risk", f"{agg.get('mean_hallucination_risk', 0):.1%}")
                c3.metric("Flagged Sentences", agg.get("n_hallucinated_sentences", 0))

            st.markdown("**🔬 Per-Sentence Analysis**")
            for i, sr in enumerate(sent_results):
                color = "#EF233C" if sr["hallucination_risk"] > 0.4 else "#06D6A0" if sr["factuality_score"] > 0.5 else "#FFB703"
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
                            padding:10px 14px;margin:4px 0;border-left:3px solid {color};">
                  <p style="color:#E6EDF3;font-size:0.88em;margin:0 0 4px;">{sr['sentence']}</p>
                  <span style="color:{color};font-size:0.8em;font-weight:600;">{sr['verdict']}</span>
                  <span style="color:#8B949E;font-size:0.8em;margin-left:10px;">
                    F:{sr['factuality_score']:.2%} · H:{sr['hallucination_risk']:.2%}
                  </span>
                </div>
                """, unsafe_allow_html=True)
