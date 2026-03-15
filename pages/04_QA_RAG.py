"""Page 04 — Question Answering / RAG"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS
from app import render_sidebar
render_sidebar()    
if "selected_example" not in st.session_state:
    st.session_state.selected_example = None
    
if "doc_len" not in st.session_state:
    st.session_state.doc_len = 3

st.set_page_config(page_title="Q&A / RAG | NEUROLEX", page_icon="💡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#06D6A0,#048A81)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea,.stTextInput>div>div>input{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
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
styled_header("💡", "Q&A / RAG", "Retrieval-Augmented Question Answering — TF-IDF retrieval + RoBERTa reader", "#06D6A0")
no_model_warning(MODELS["qa"]["model_name"])

DEFAULT_DOCS = """The Large Hadron Collider (LHC) is the world's largest and most powerful particle collider,
built by CERN between 1998 and 2008. It lies in a tunnel 27 km in circumference beneath the France–Switzerland border.
The Higgs boson was discovered at the LHC in 2012, confirming the Standard Model of particle physics.

Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability
and allows programmers to express concepts in fewer lines of code. Python supports multiple programming paradigms,
including structured, object-oriented, and functional programming.

The Amazon rainforest covers over 5.5 million square kilometers and is home to 10% of the world's species.
It produces 20% of the world's oxygen and plays a crucial role in regulating the global climate.
Deforestation threatens approximately 17% of the forest in the past 50 years."""

EXAMPLE_QUESTIONS = [
    "What is the LHC and when was it built?",
    "Who created Python and when?",
    "What percentage of oxygen does the Amazon produce?",
]

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("**📚 Document Corpus**")
    docs_text = st.text_area("Documents", value=DEFAULT_DOCS, height=200, label_visibility="collapsed")

with col2:
    st.markdown("**❓ Question**")
    qcols = st.columns(1)

    for i, eq in enumerate(EXAMPLE_QUESTIONS):
        if st.button(f"🔹 {eq[:45]}…", key=f"qa_ex_{i}", use_container_width=True):
            st.session_state.selected_example = eq

    question = st.text_input(
        "Your question", value=st.session_state.selected_example or "",
        placeholder="Ask anything about the documents...",
    )
    docs = [d.strip() for d in docs_text.strip().split("\n\n") if d.strip()]
    st.session_state.doc_len = max(1,len(docs))
    top_k = st.slider("Retrieved Passages (top-k)", 1, st.session_state.doc_len, 3)
    run = st.button("💡 Get Answer", use_container_width=True)

if run:
    if not docs_text.strip():
        st.error("Please provide documents.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        if not docs:
            docs = [docs_text]

        with st.spinner("Retrieving context and extracting answer..."):
            from neurolex.modules.rag_qa import RAGQuestionAnswerer
            rag = RAGQuestionAnswerer()
            n_chunks = rag.index_documents(docs)
            result = rag.answer(question, top_k=top_k)

        st.markdown("---")

        # Answer card
        answer = result.get("answer", "No answer found.")
        score = result.get("score", 0.0)
        color = "#06D6A0" if score > 0.5 else "#FFB703" if score > 0.2 else "#EF233C"

        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:14px;
                    padding:20px;margin-bottom:16px;border-top:3px solid {color};">
          <p style="color:#8B949E;font-size:0.8em;margin:0;">💬 ANSWER</p>
          <h2 style="color:{color};font-size:1.4em;margin:8px 0;">{answer}</h2>
          <div style="display:flex;gap:12px;margin-top:10px;">
            <span style="background:{color}22;color:{color};border-radius:8px;
                         padding:3px 12px;font-size:0.85em;font-weight:600;">
              Confidence: {score:.1%}
            </span>
            <span style="background:#21262D;color:#8B949E;border-radius:8px;
                         padding:3px 12px;font-size:0.85em;">
              Chunks indexed: {n_chunks}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("🎯 Confidence", f"{score:.2%}")
        c2.metric("📄 Chunks Indexed", n_chunks)

        with st.expander("📄 Retrieved Context Passages", expanded=True):
            for chunk_info in result.get("retrieved_chunks", []):
                relevance = chunk_info["score"]
                rc = "#06D6A0" if relevance > 0.3 else "#48CAE4"
                st.markdown(f"""
                <div style="background:#21262D;border-radius:8px;padding:12px;margin:6px 0;
                            border-left:3px solid {rc};">
                  <span style="color:{rc};font-size:0.8em;font-weight:600;">
                    Relevance: {relevance:.3f}
                  </span>
                  <p style="color:#E6EDF3;font-size:0.88em;margin:6px 0 0;">{chunk_info['chunk'][:300]}…</p>
                </div>
                """, unsafe_allow_html=True)
