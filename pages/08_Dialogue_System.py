"""Page 08 — Conversational Dialogue System"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurolex.utils import styled_header, no_model_warning
from neurolex.config import MODELS
from app import render_sidebar
render_sidebar()    
st.set_page_config(page_title="Dialogue System | NEUROLEX", page_icon="💬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0D1117!important;color:#E6EDF3!important;}
[data-testid="stSidebar"]{background:#161B22!important;border-right:1px solid #30363D!important;}
.stButton>button{background:linear-gradient(135deg,#F77F00,#D62246)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextInput>div>div>input{background:#21262D!important;border:1px solid #30363D!important;border-radius:8px!important;color:#E6EDF3!important;}
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
styled_header("💬", "Dialogue System", "Multi-turn conversational AI using DialoGPT-medium with session memory", "#F77F00")
no_model_warning(MODELS["dialogue"]["model_name"])

# Session state init
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []  # list of (role, text)
if "dialogue_model" not in st.session_state:
    st.session_state.dialogue_model = None

st.markdown("""
<div style="background:#161B22;border:1px solid #30363D;border-radius:10px;padding:12px 16px;margin-bottom:14px;">
  <p style="color:#8B949E;font-size:0.82em;margin:0;">
    💡 <b style="color:#F77F00;">DialoGPT-medium</b> is a GPT-2 model fine-tuned on 147M Reddit conversations.
    It maintains multi-turn context in-session. Click <b>Clear Chat</b> to reset memory.
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.dialogue_history = []
        st.session_state.dialogue_model = None
        st.rerun()

# ─── Chat history display ────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.dialogue_history:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#8B949E;">
          <div style="font-size:3em;">💬</div>
          <p>Start a conversation! Type a message below.</p>
          <p style="font-size:0.85em;">Suggested openers: "Hello!", "Tell me a joke", "What do you think about space exploration?"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for role, text in st.session_state.dialogue_history:
            if role == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:6px 0;">
                  <div style="background:#6C63FF;color:white;border-radius:18px 18px 4px 18px;
                              padding:10px 16px;max-width:70%;font-size:0.92em;line-height:1.5;">
                    {text}
                  </div>
                  <span style="font-size:1.2em;margin-left:8px;align-self:flex-end;">👤</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;margin:6px 0;">
                  <span style="font-size:1.2em;margin-right:8px;align-self:flex-end;">🤖</span>
                  <div style="background:#21262D;color:#E6EDF3;border-radius:18px 18px 18px 4px;
                              padding:10px 16px;max-width:70%;font-size:0.92em;line-height:1.5;
                              border:1px solid #30363D;">
                    {text}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ─── Input ──────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message...",
            label_visibility="collapsed",
        )
    with c2:
        send = st.form_submit_button("Send ➤", use_container_width=True)

if send and user_input.strip():
    from neurolex.modules.dialogue import DialogueSystem

    # Init model once per session
    if st.session_state.dialogue_model is None:
        with st.spinner("Loading DialoGPT-medium (first time only)..."):
            st.session_state.dialogue_model = DialogueSystem()

    dialogue = st.session_state.dialogue_model
    st.session_state.dialogue_history.append(("user", user_input))

    with st.spinner("Generating response..."):
        response = dialogue.chat(user_input)

    st.session_state.dialogue_history.append(("bot", response))
    st.rerun()

# Status bar
if st.session_state.dialogue_history:
    n_turns = len(st.session_state.dialogue_history) // 2
    st.markdown(f"""
    <div style="text-align:center;color:#8B949E;font-size:0.78em;margin-top:8px;">
      Conversation turns: {n_turns} · Model: DialoGPT-medium · Context: last ~900 tokens
    </div>
    """, unsafe_allow_html=True)
