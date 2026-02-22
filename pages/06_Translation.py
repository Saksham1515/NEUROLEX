"""Page 06 — Multilingual Machine Translation"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import TRANSLATION_PAIRS

st.set_page_config(page_title="Translation | NEUROLEX", page_icon="🌍", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#4CC9F0,#4361EE)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextArea>div>div>textarea{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
.stSelectbox>div>div{background:#21262D!important;border:1px solid #30363D!important;color:#E6EDF3!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

styled_header("🌍", "Machine Translation", "Multilingual translation via Helsinki-NLP OPUS-MT (MarianMT), 14+ language pairs", "#4CC9F0")
st.info("💡 Each language pair uses a separately trained MarianMT model (~300MB). The first run may take time to download.", icon="ℹ️")

EXAMPLES = {
    "English → French": "Artificial intelligence is transforming the way we live and work.",
    "English → German": "The future of sustainable energy lies in renewable sources like solar and wind.",
    "English → Spanish": "Machine learning enables computers to learn from data without being explicitly programmed.",
    "English → Hindi": "Technology has made communication faster and more accessible worldwide.",
}

col1, col2 = st.columns([1, 1])
with col1:
    pair = st.selectbox("🌐 Language Pair", list(TRANSLATION_PAIRS.keys()), index=0)
    max_len = st.slider("Max Output Length (tokens)", 64, 512, 256)

with col2:
    st.markdown("**Quick Examples:**")
    for pair_name, ex_text in list(EXAMPLES.items())[:3]:
        if st.button(f"📝 {pair_name}", key=f"t_ex_{pair_name[:6]}", use_container_width=True):
            st.session_state["trans_pair"] = pair_name
            st.session_state["trans_text"] = ex_text

col_l, col_r = st.columns(2)
with col_l:
    st.markdown(f"**📥 Source Text** ({TRANSLATION_PAIRS[pair][0].upper()})")
    prefill = st.session_state.get("trans_text", "")
    source_text = st.text_area(
        "Source", value=prefill, height=180,
        placeholder="Enter text to translate...", label_visibility="collapsed",
    )

run = st.button("🌍 Translate", use_container_width=False)

if run:
    if not source_text.strip():
        st.warning("Please enter text to translate.")
    else:
        with st.spinner(f"Translating ({pair})..."):
            from neurolex.modules.translator import MultilingualTranslator
            translator = MultilingualTranslator()
            result = translator.translate(source_text, pair_name=pair, max_length=max_len)

        if "error" in result:
            st.error(f"❌ Translation Error: {result['error']}")
        else:
            with col_r:
                st.markdown(f"**📤 Translation** ({result['target_lang'].upper()})")
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #4CC9F0;border-radius:10px;
                            padding:16px;min-height:180px;font-size:1em;line-height:1.7;
                            color:#E6EDF3;">
                  {result['translation']}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Source Language", result["source_lang"].upper())
            c2.metric("Target Language", result["target_lang"].upper())
            c3.metric("Source Chars", result["source_chars"])
            c4.metric("Output Chars", result["output_chars"])

            bleu = translator.bleu_estimate(source_text, result["translation"])
            st.markdown(f"""
            <div style="background:#21262D;border-radius:8px;padding:10px 14px;margin-top:8px;
                        display:inline-block;">
              <span style="color:#8B949E;font-size:0.85em;">Lexical overlap (BLEU-1 estimate): </span>
              <b style="color:#4CC9F0;">{bleu:.3f}</b>
              <span style="color:#8B949E;font-size:0.8em;"> (low overlap expected for cross-lingual translation)</span>
            </div>
            """, unsafe_allow_html=True)
