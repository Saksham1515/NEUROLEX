"""Page 07 — Topic Modeling & Trend Detection"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header

st.set_page_config(page_title="Topic Modeling | NEUROLEX", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#7209B7,#6C63FF)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("📊", "Topic Modeling", "Neural topic discovery using TF-IDF + SVD + K-Means (BERTopic-inspired)", "#7209B7")

DEFAULT_CORPUS = """AI is transforming the healthcare industry with machine learning models for diagnosis.
Deep learning models achieve superhuman performance in medical imaging and radiology.
Scientists discovered a new exoplanet orbiting a nearby star in the habitable zone.
NASA's James Webb Space Telescope has captured images of distant galaxies and nebulae.
Climate change is causing unprecedented flooding and wildfires across the globe.
Renewable energy sources like solar panels are becoming cheaper and more efficient.
The stock market experienced volatility amid inflation fears and central bank rate hikes.
Cryptocurrency markets crashed as Bitcoin fell below $20,000 for the first time in two years.
Premier League clubs are spending record transfer fees to attract top football talent.
The Olympics saw athletes break multiple world records in swimming and track events.
Quantum computers achieved a new milestone by solving problems impossible for classical machines.
A new programming language gained popularity for its performance and type safety features.
Electric vehicles are outselling gasoline cars in Norway and several European markets.
Autonomous driving technology faces regulatory hurdles despite technological progress."""

st.markdown("**📄 Document Collection** (one document per line — paste news headlines, tweets, articles, etc.)")
corpus_text = st.text_area("Corpus", value=DEFAULT_CORPUS, height=200, label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    n_topics = st.slider("Number of Topics", 2, 10, 5)
with col2:
    n_keywords = st.slider("Keywords per Topic", 4, 12, 8)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("📊 Discover Topics", use_container_width=True)

if run:
    docs = [d.strip() for d in corpus_text.strip().split("\n") if len(d.strip()) > 10]
    if len(docs) < 3:
        st.warning("Please provide at least 3 documents.")
    else:
        with st.spinner(f"Discovering {n_topics} topics in {len(docs)} documents..."):
            from neurolex.modules.topic_modeler import TopicModeler
            modeler = TopicModeler(n_topics=n_topics, n_keywords=n_keywords)
            result = modeler.fit(docs)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("📄 Documents", result["n_docs"])
        c2.metric("🏷️ Topics Found", result["n_topics"])

        TOPIC_COLORS = ["#6C63FF", "#F72585", "#06D6A0", "#FFB703", "#4CC9F0",
                        "#7209B7", "#EF233C", "#48CAE4", "#F77F00", "#2EC4B6"]

        tab1, tab2, tab3 = st.tabs(["🏷️ Topic Keywords", "📊 Topic Distribution", "📄 Document Assignments"])

        with tab1:
            topic_cols = st.columns(min(n_topics, 3))
            for i, topic in enumerate(result["topics"]):
                color = TOPIC_COLORS[i % len(TOPIC_COLORS)]
                with topic_cols[i % 3]:
                    kw_html = " ".join([
                        f'<span style="background:{color}33;color:{color};'
                        f'border-radius:12px;padding:2px 10px;font-size:0.82em;'
                        f'font-weight:600;margin:2px;display:inline-block;">{kw}</span>'
                        for kw in topic["keywords"]
                    ])
                    st.markdown(f"""
                    <div style="background:#161B22;border:1px solid #30363D;border-radius:12px;
                                padding:14px;margin-bottom:10px;border-top:3px solid {color};">
                      <b style="color:{color};">{topic['label']}</b>
                      <p style="color:#8B949E;font-size:0.8em;margin:4px 0;">{topic['doc_count']} documents</p>
                      <div style="margin-top:8px;">{kw_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            topics_sorted = modeler.trend_scores()
            fig = go.Figure([go.Bar(
                x=[t["label"] for t in topics_sorted],
                y=[t["doc_count"] for t in topics_sorted],
                marker_color=TOPIC_COLORS[:len(topics_sorted)],
                text=[t["doc_count"] for t in topics_sorted],
                textposition="outside",
                hovertext=[t["keywords_str"] for t in topics_sorted],
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(gridcolor="#30363D"),
                yaxis=dict(gridcolor="#30363D", title="Document Count"),
                margin=dict(l=10, r=10, t=20, b=10), height=350,
                title=dict(text="Topic Prevalence (Trend Detection)", font=dict(color="#E6EDF3")),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            fig2 = go.Figure(go.Pie(
                labels=[t["label"] for t in topics_sorted],
                values=[t["doc_count"] for t in topics_sorted],
                marker=dict(colors=TOPIC_COLORS[:len(topics_sorted)]),
                hole=0.5,
                textinfo="label+percent",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3", family="Inter"),
                margin=dict(l=0, r=0, t=0, b=0), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            assignments = result["document_assignments"]
            for i, (doc, tid) in enumerate(zip(docs, assignments)):
                topic = result["topics"][tid]
                color = TOPIC_COLORS[tid % len(TOPIC_COLORS)]
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;
                            padding:10px 14px;margin:4px 0;border-left:3px solid {color};">
                  <span style="color:{color};font-size:0.8em;font-weight:600;">{topic['label']}</span>
                  <p style="color:#E6EDF3;font-size:0.88em;margin:4px 0 0;">{doc[:120]}{'…' if len(doc)>120 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
