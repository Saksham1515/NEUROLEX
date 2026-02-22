"""Page 03 — Semantic Search & Information Retrieval"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS

st.set_page_config(page_title="Semantic Search | NEUROLEX", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#F72585,#7209B7)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea,.stTextInput>div>div>input{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("🔎", "Semantic Search", "Dense vector retrieval using sentence-transformers all-MiniLM-L6-v2", "#F72585")
no_model_warning(MODELS["semantic_search"]["model_name"])

DEFAULT_CORPUS = """Artificial intelligence is transforming healthcare with early disease detection.
Machine learning models can predict stock market trends with high accuracy.
Climate change is accelerating global warming and leading to extreme weather events.
Python is the most popular programming language for data science and AI research.
Quantum computing promises to solve problems intractable for classical computers.
Space exploration has revealed new exoplanets in habitable zones around distant stars.
Electric vehicles are reducing carbon emissions and reshaping the automotive industry.
Deep learning neural networks have achieved superhuman performance in image recognition.
Natural language processing enables computers to understand and generate human text.
Blockchain technology provides decentralized and secure transaction ledgers."""

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("**📚 Document Corpus** (one document per line)")
    corpus_text = st.text_area("Corpus", value=DEFAULT_CORPUS, height=220, label_visibility="collapsed")
    top_k = st.slider("Top-K Results", 1, 10, 5)

with col2:
    st.markdown("**🔍 Query**")
    EXAMPLE_QUERIES = [
        "How is AI used in medicine?",
        "What is happening with the environment?",
        "Tell me about space and planets",
    ]
    qcols = st.columns(3)
    selected_q = None
    for i, (qc, eq) in enumerate(zip(qcols, EXAMPLE_QUERIES)):
        with qc:
            if st.button(f"Q{i+1}", key=f"ss_q_{i}", use_container_width=True):
                selected_q = eq

    query = st.text_input("Search query", value=selected_q or "", placeholder="Enter your semantic query...")

    c1, c2 = st.columns(2)
    with c1:
        run_search = st.button("🔍 Search", use_container_width=True)
    with c2:
        show_matrix = st.button("🗂️ Similarity Matrix", use_container_width=True)

if run_search and query.strip():
    docs = [d.strip() for d in corpus_text.strip().split("\n") if d.strip()]
    if not docs:
        st.error("Please add documents to the corpus.")
    else:
        with st.spinner("Building index and searching..."):
            from neurolex.modules.semantic_search import SemanticSearchEngine
            engine = SemanticSearchEngine()
            engine.index_corpus(docs)
            results = engine.search(query, top_k=top_k)

        st.markdown("---")
        st.markdown(f"**Found {len(results)} results for:** *{query}*")

        for res in results:
            score = res["score"]
            bar_width = int(score * 100)
            color = "#F72585" if score > 0.7 else "#6C63FF" if score > 0.4 else "#8B949E"
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
                        padding:14px;margin:8px 0;border-left:4px solid {color};">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:{color};font-weight:700;font-size:0.9em;">Rank #{res['rank']}</span>
                <span style="background:#21262D;color:{color};border-radius:12px;
                             padding:2px 10px;font-size:0.85em;font-weight:600;">
                  {score:.3f}
                </span>
              </div>
              <p style="color:#E6EDF3;font-size:0.9em;margin:0;">{res['snippet']}</p>
              <div style="background:#21262D;border-radius:4px;height:4px;margin-top:8px;">
                <div style="width:{bar_width}%;height:100%;background:{color};border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

if show_matrix:
    docs = [d.strip() for d in corpus_text.strip().split("\n") if d.strip()][:8]
    if len(docs) < 2:
        st.warning("Add at least 2 documents to see the similarity matrix.")
    else:
        with st.spinner("Computing similarity matrix..."):
            from neurolex.modules.semantic_search import SemanticSearchEngine
            engine = SemanticSearchEngine()
            matrix = engine.get_similarity_matrix(docs)

        labels = [d[:35] + "…" for d in docs]
        fig = go.Figure(go.Heatmap(
            z=matrix, x=labels, y=labels,
            colorscale=[[0, "#161B22"], [0.5, "#6C63FF"], [1, "#F72585"]],
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3", family="Inter", size=9),
            margin=dict(l=10, r=10, t=30, b=10), height=450,
            title=dict(text="Pairwise Cosine Similarity Matrix", font=dict(color="#E6EDF3")),
        )
        st.plotly_chart(fig, use_container_width=True)
