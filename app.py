"""
NEUROLEX — Main Streamlit Application Entry Point
"""
import streamlit as st
st.markdown("""
<style>
/* Hide Streamlit auto multipage navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEUROLEX | Advanced NLP Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NEUROLEX — End-to-End Advanced NLP System v1.0",
    },
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

  /* Root variables */
  :root {
    --primary: #6C63FF;
    --secondary: #48CAE4;
    --accent: #F72585;
    --success: #06D6A0;
    --warning: #FFB703;
    --danger: #EF233C;
    --bg: #0D1117;
    --surface: #161B22;
    --surface2: #21262D;
    --border: #30363D;
    --text: #E6EDF3;
    --muted: #8B949E;
  }

  /* Base */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Main area */
  .main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--primary), #4ECDC4) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
  }

  /* Text areas */
  .stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95em !important;
  }
  .stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
  }

  /* Select boxes */
  .stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  /* Sliders */
  .stSlider > div { color: var(--text) !important; }

  /* Metrics */
  [data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }
  [data-testid="stMetricValue"] { color: var(--primary) !important; }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
  }

  /* Info/warning boxes */
  .stAlert { border-radius: 10px !important; }

  /* Hide default Streamlit elements */
  #MainMenu, footer { visibility: hidden; }

  /* Plotly charts */
  .js-plotly-plot { border-radius: 12px !important; }

  /* Code blocks */
  code {
    background: var(--surface2) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--secondary) !important;
  }
  
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo & title
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px;">
          <div style="font-size:3em;">🧠</div>
          <h1 style="font-size:1.6em;font-weight:900;
                     background:linear-gradient(135deg,#6C63FF,#48CAE4);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     margin:4px 0;">NEUROLEX</h1>
          <p style="color:#8B949E;font-size:0.78em;margin:0;">
            End-to-End NLP Platform v1.0
          </p>
        </div>
        <hr style="border-color:#30363D;margin:8px 0 16px;">
        """, unsafe_allow_html=True)

        st.markdown("**🚀 NLP Modules**")
        st.page_link("app.py", label="🏠 Home & Architecture", icon=None)
        st.markdown("<hr style='border-color:#30363D;margin:6px 0;'>", unsafe_allow_html=True)
        st.page_link("pages/01_Text_Classification.py", label="🏷️ Text Classification")
        st.page_link("pages/02_NER_Entity_Linking.py", label="🔍 NER & Entity Linking")
        st.page_link("pages/03_Semantic_Search.py", label="🔎 Semantic Search")
        st.page_link("pages/04_QA_RAG.py", label="💡 Q&A / RAG")
        st.page_link("pages/05_Summarization.py", label="📝 Summarization")
        st.page_link("pages/06_Translation.py", label="🌍 Translation")
        st.page_link("pages/07_Topic_Modeling.py", label="📊 Topic Modeling")
        st.page_link("pages/08_Dialogue_System.py", label="💬 Dialogue System")
        st.page_link("pages/09_Hallucination_Detection.py", label="⚠️ Hallucination Detection")
        st.page_link("pages/10_Bias_Toxicity.py", label="🛡️ Bias & Toxicity")

        st.markdown("<hr style='border-color:#30363D;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75em;color:#8B949E;padding:4px 0;">
          <b style="color:#E6EDF3;">Stack:</b> Python · Streamlit · HuggingFace<br>
          <b style="color:#E6EDF3;">Device:</b> CPU / CUDA (auto-detect)<br>
          <b style="color:#E6EDF3;">Models:</b> BART · BERT · RoBERTa · GPT-2
        </div>
        """, unsafe_allow_html=True)


render_sidebar()

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;">
  <h1 style="font-size:3.2em;font-weight:900;
             background:linear-gradient(135deg,#6C63FF,#48CAE4,#F72585);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:0;">🧠 NEUROLEX</h1>
  <p style="color:#8B949E;font-size:1.2em;margin:10px 0 20px;">
    End-to-End Advanced NLP Research Platform
  </p>
  <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
    <span style="background:#6C63FF22;color:#6C63FF;border:1px solid #6C63FF44;
                 padding:4px 14px;border-radius:20px;font-size:0.85em;font-weight:600;">
      10 NLP Tasks
    </span>
    <span style="background:#48CAE422;color:#48CAE4;border:1px solid #48CAE444;
                 padding:4px 14px;border-radius:20px;font-size:0.85em;font-weight:600;">
      HuggingFace Transformers
    </span>
    <span style="background:#F7258522;color:#F72585;border:1px solid #F7258544;
                 padding:4px 14px;border-radius:20px;font-size:0.85em;font-weight:600;">
      Production Ready
    </span>
    <span style="background:#06D6A022;color:#06D6A0;border:1px solid #06D6A044;
                 padding:4px 14px;border-radius:20px;font-size:0.85em;font-weight:600;">
      Explainability Built-in
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:12px;padding:16px;">
      <h4 style="color:#6C63FF;margin:0 0 12px;">📐 Design Principles</h4>
      <div style="display:flex;flex-direction:column;gap:8px;">
    """, unsafe_allow_html=True)

principles = [
        ("🔧", "Modular", "Each task is an isolated, reusable class"),
        ("📦", "Cached", "Models loaded once via @st.cache_resource"),
        ("🔬", "Explainable", "Attribution scores, entity highlights"),
        ("🔄", "Active Learning", "Uncertainty-based sample selection"),
        ("🌐", "Multilingual", "14+ language translation pairs"),
        ("🏭", "Production-ready", "Config-driven, extensible pipeline"),
    ]
for icon, title, desc in principles:
        st.markdown(f"""
        <div style="background:#21262D;border-radius:8px;padding:10px 12px;margin:4px 0;">
          <span style="font-size:1.1em;">{icon}</span>
          <span style="font-weight:700;color:#E6EDF3;margin-left:6px;">{title}</span>
          <p style="color:#8B949E;font-size:0.8em;margin:2px 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# ─── Model Selection Table ────────────────────────────────────────────────────
st.markdown("## 🤖 Model Selection")
import pandas as pd
import plotly.graph_objects as go

models_data = {
    "Module": [
        "Text Classification", "NER", "Entity Linking",
        "Semantic Search", "Q&A / RAG", "Summarization",
        "Translation", "Topic Modeling", "Dialogue",
        "Hallucination", "Toxicity",
    ],
    "Model": [
        "facebook/bart-large-mnli", "dslim/bert-base-NER", "Wikipedia REST API",
        "all-MiniLM-L6-v2", "deepset/roberta-base-squad2", "facebook/bart-large-cnn",
        "Helsinki-NLP/opus-mt-*", "TF-IDF + SVD + KMeans", "microsoft/DialoGPT-medium",
        "facebook/bart-large-mnli", "unitary/toxic-bert",
    ],
    "Approach": [
        "Zero-shot NLI", "Token Classification", "API + Linking",
        "Dense Embeddings", "Extractive QA", "Seq2Seq Generation",
        "MarianMT", "LSA + Clustering", "Causal LM",
        "NLI Entailment", "Multi-label BERT",
    ],
    "Benchmark": [
        "MNLI (90.7%)", "CoNLL-03 (F1: 91.3)", "—",
        "BEIR / MS MARCO", "SQuAD2 (F1: 84.2)", "CNN/DM (ROUGE-L: 40.9)",
        "WMT (BLEU varies)", "Topic Coherence CV", "DailyDialog PPL",
        "FactCC / FEVER", "Jigsaw AUC: 0.98",
    ],
}

df_models = pd.DataFrame(models_data)
fig = go.Figure(
    data=[go.Table(
        columnwidth=[120, 160, 130, 170],
        header=dict(
            values=["<b>Module</b>", "<b>Model</b>", "<b>Approach</b>", "<b>Benchmark</b>"],
            fill_color="#6C63FF",
            font=dict(color="white", size=12, family="Inter"),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[df_models[c] for c in df_models.columns],
            fill_color=[["#21262D", "#161B22"] * 10],
            font=dict(color="#E6EDF3", size=11, family="Inter"),
            align="left",
            height=30,
        ),
    )]
)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    height=360,
)
st.plotly_chart(fig, use_container_width=True)

# ─── Module Grid ──────────────────────────────────────────────────────────────
st.markdown("## 🎯 NLP Modules")
from neurolex.config import MODULE_INFO

cols = st.columns(3)
for i, mod in enumerate(MODULE_INFO):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #30363D;border-radius:12px;
                    padding:16px;margin-bottom:12px;
                    border-top:3px solid {mod['color']};
                    transition:all 0.2s;">
          <div style="font-size:1.8em;margin-bottom:6px;">{mod['icon']}</div>
          <h4 style="color:{mod['color']};margin:0;font-size:0.95em;">{mod['name']}</h4>
          <p style="color:#8B949E;font-size:0.8em;margin:4px 0 0;">{mod['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;color:#8B949E;font-size:0.8em;">
  <hr style="border-color:#30363D;margin-bottom:16px;">
  🧠 <b style="color:#6C63FF;">NEUROLEX</b> — Advanced NLP Research Platform · Built with
  ❤️(saksham) using Python, Streamlit & HuggingFace Transformers
</div>
""", unsafe_allow_html=True)
