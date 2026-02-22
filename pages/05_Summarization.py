"""Page 05 — Abstractive & Extractive Summarization"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS

st.set_page_config(page_title="Summarization | NEUROLEX", page_icon="📝", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#FFB703,#F77F00)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("📝", "Summarization", "Abstractive (BART-CNN) & Extractive (TF-IDF sentence scoring) summarization", "#FFB703")
no_model_warning(MODELS["summarizer_abstractive"]["model_name"])

EXAMPLE_TEXT = """Artificial intelligence is rapidly transforming every sector of the modern economy.
From healthcare to transportation, AI-powered systems are automating complex tasks, improving
decision-making accuracy, and enabling entirely new categories of products and services.
In the medical field, machine learning algorithms can now detect cancers in radiology images
with accuracy rivaling or surpassing experienced radiologists. In autonomous vehicles,
deep learning models process sensor data in real time to navigate complex environments safely.
Financial institutions are deploying AI for fraud detection, algorithmic trading, and personalized
financial advice. However, the rise of AI also raises profound questions about employment,
privacy, algorithmic bias, and the concentration of power in the hands of a few large corporations.
Policymakers around the world are grappling with how to regulate this powerful technology while
not stifling innovation. The stakes could not be higher: how we govern AI in the coming years
may determine the shape of human civilization for decades to come."""

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**📄 Document to Summarize**")
    if st.button("📰 Load Example Article", key="sum_ex"):
        text = EXAMPLE_TEXT
    else:
        text = None
    document = st.text_area(
        "Document", value=text or "", height=220,
        placeholder="Paste a long document here...", label_visibility="collapsed",
    )

with col2:
    st.markdown("**⚙️ Settings**")
    mode = st.radio("Summarization Mode", ["✨ Abstractive", "✂️ Extractive", "📊 Both"], index=2)
    ratio = st.slider("Summary Length Ratio", 0.1, 0.6, 0.3, 0.05,
                      help="Target summary as fraction of input length")
    n_sentences = st.slider("Extractive Sentences", 2, 8, 4)
    run = st.button("📝 Summarize", use_container_width=True)

if run:
    if not document.strip():
        st.warning("Please enter a document to summarize.")
    else:
        from neurolex.modules.summarizer import DocumentSummarizer
        summarizer = DocumentSummarizer()

        st.markdown("---")

        if "Abstractive" in mode or "Both" in mode:
            with st.spinner("Generating abstractive summary (BART-CNN)..."):
                abs_result = summarizer.summarize_abstractive(document, ratio=ratio)
            c1, c2, c3 = st.columns(3)
            c1.metric("📥 Input Words", abs_result["word_count_in"])
            c2.metric("📤 Output Words", abs_result["word_count_out"])
            c3.metric("📊 Compression", f"{abs_result['compression_ratio']:.1%}")

            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #FFB703;border-radius:12px;padding:18px;">
              <p style="color:#FFB703;font-size:0.85em;font-weight:600;margin:0 0 8px;">
                ✨ ABSTRACTIVE SUMMARY (BART-CNN)
              </p>
              <p style="color:#E6EDF3;font-size:1em;line-height:1.7;margin:0;">{abs_result['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

        if "Extractive" in mode or "Both" in mode:
            with st.spinner("Extracting key sentences..."):
                ext_result = summarizer.summarize_extractive(document, n_sentences=n_sentences)

            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #48CAE4;border-radius:12px;
                        padding:18px;margin-top:14px;">
              <p style="color:#48CAE4;font-size:0.85em;font-weight:600;margin:0 0 8px;">
                ✂️ EXTRACTIVE SUMMARY (TF-IDF Sentence Scoring)
              </p>
              <p style="color:#E6EDF3;font-size:1em;line-height:1.7;margin:0;">{ext_result['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            if ext_result.get("sentence_scores"):
                with st.expander("📊 Sentence Importance Scores"):
                    top_sents = ext_result["sentence_scores"][:8]
                    fig = go.Figure(go.Bar(
                        x=[s["score"] for s in top_sents],
                        y=[f"Sent {i+1}: {s['sentence'][:40]}…" for i, s in enumerate(top_sents)],
                        orientation="h",
                        marker_color="#48CAE4",
                        text=[f"{s['score']:.3f}" for s in top_sents],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
                        font=dict(color="#E6EDF3", family="Inter"),
                        yaxis=dict(autorange="reversed", gridcolor="#30363D"),
                        xaxis=dict(gridcolor="#30363D"),
                        margin=dict(l=10, r=60, t=10, b=10), height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if "Both" in mode and abs_result and ext_result:
            with st.expander("📏 ROUGE Estimate (Abstractive vs Extractive)"):
                rouge = summarizer.rouge_estimate(ext_result["summary"], abs_result["summary"])
                c1, c2 = st.columns(2)
                c1.metric("ROUGE-1", f"{rouge['rouge_1']:.3f}")
                c2.metric("ROUGE-2", f"{rouge['rouge_2']:.3f}")
