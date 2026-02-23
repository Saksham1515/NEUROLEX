"""Page 02 — Named Entity Recognition & Entity Linking"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS
from app import render_sidebar
render_sidebar()    
st.set_page_config(page_title="NER & Entity Linking | NEUROLEX", page_icon="🔍", layout="wide")
st.markdown("""
<style>
/* Hide Streamlit auto multipage navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#48CAE4,#6C63FF)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("🔍", "NER & Entity Linking", "BERT-NER extraction + Wikipedia entity linking", "#48CAE4")
no_model_warning(MODELS["ner"]["model_name"])

EXAMPLES = [
    "Elon Musk announced that Tesla will open a new Gigafactory in Berlin, Germany next year.",
    "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
    "OpenAI, founded in San Francisco, released GPT-4 as a breakthrough in artificial intelligence.",
]

eg_cols = st.columns(3)
selected = None
for i, (ec, ex) in enumerate(zip(eg_cols, EXAMPLES)):
    with ec:
        if st.button(f"Example {i+1}", key=f"ner_ex_{i}", use_container_width=True):
            selected = ex

text = st.text_area(
    "Input Text", value=selected or "",
    height=150, placeholder="Paste text with named entities...",
)
link_entities = st.checkbox("🔗 Enable Wikipedia Entity Linking", value=True)
run = st.button("🚀 Extract Entities", use_container_width=False)

if run and text.strip():
    with st.spinner("Extracting named entities..."):
        from neurolex.modules.ner import NERLinker
        ner = NERLinker()
        entities = ner.extract_entities(text)
        stats = ner.get_entity_stats(entities)

    st.markdown("---")

    if not entities:
        st.info("No named entities found in the text.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entities", len(entities))
        c2.metric("Persons (PER)", stats.get("PER", 0))
        c3.metric("Organizations (ORG)", stats.get("ORG", 0))
        c4.metric("Locations (LOC)", stats.get("LOC", 0))

        tab1, tab2, tab3 = st.tabs(["📝 Annotated Text", "📊 Entity Chart", "🔗 Wikipedia Links"])

        with tab1:
            annotated = ner.annotate_html(text, entities)
            st.markdown(annotated, unsafe_allow_html=True)
            st.markdown("<br>**Legend:**", unsafe_allow_html=True)
            leg_cols = st.columns(4)
            legend = [("PER", "#6C63FF", "Person"), ("ORG", "#F72585", "Organization"),
                      ("LOC", "#06D6A0", "Location"), ("MISC", "#FFB703", "Miscellaneous")]
            for col, (etype, color, label) in zip(leg_cols, legend):
                with col:
                    st.markdown(
                        f'<span style="background:{color}33;color:{color};'
                        f'border-radius:4px;padding:3px 10px;font-size:0.85em;font-weight:600;">'
                        f'{etype}: {label}</span>',
                        unsafe_allow_html=True,
                    )

        with tab2:
            entity_types = list(stats.keys())
            entity_counts = list(stats.values())
            colors = {"PER": "#6C63FF", "ORG": "#F72585", "LOC": "#06D6A0", "MISC": "#FFB703"}
            bar_colors = [colors.get(t, "#8B949E") for t in entity_types]

            fig = go.Figure([
                go.Bar(
                    x=entity_types, y=entity_counts,
                    marker_color=bar_colors,
                    text=entity_counts, textposition="outside",
                )
            ])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3", family="Inter"),
                xaxis=dict(gridcolor="#30363D"),
                yaxis=dict(gridcolor="#30363D"),
                margin=dict(l=10, r=10, t=20, b=10), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**All Entities:**")
            import pandas as pd
            df = pd.DataFrame([{"Entity": e["word"], "Type": e["entity"], "Confidence": f"{e['score']:.2%}"} for e in entities])
            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab3:
            if link_entities:
                unique_entities = list({e["word"]: e for e in entities}.values())[:6]
                for ent in unique_entities:
                    with st.spinner(f"Linking {ent['word']}..."):
                        linked = ner.link_entity(ent["word"])
                    color = ent["color"]
                    found_badge = "✅ Found" if linked["found"] else "❌ Not Found"
                    st.markdown(f"""
                    <div style="background:#161B22;border:1px solid #30363D;border-radius:10px;
                                padding:14px;margin:8px 0;border-left:4px solid {color};">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <b style="color:{color};font-size:1em;">{ent['word']}</b>
                        <span style="background:#21262D;color:#8B949E;border-radius:12px;
                                     padding:2px 10px;font-size:0.8em;">{ent['entity']}</span>
                        <span style="font-size:0.8em;color:#06D6A0;">{found_badge}</span>
                      </div>
                      <p style="color:#E6EDF3;font-size:0.85em;margin:8px 0 4px;">{linked['summary']}</p>
                      {"<a href='" + linked['url'] + "' target='_blank' style='color:#48CAE4;font-size:0.8em;'>→ Wikipedia</a>" if linked.get('url') else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable Wikipedia Entity Linking above to see linked articles.")

elif run:
    st.warning("Please enter text to analyze.")
