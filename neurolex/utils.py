"""
NEUROLEX — Shared Utilities
Text preprocessing, timing, and Streamlit helpers.
"""
import re
import time
import functools
import streamlit as st
from typing import Any, Callable


# ─── Text Preprocessing ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize text."""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def truncate_text(text: str, max_chars: int = 1024) -> str:
    """Truncate text to a maximum character count."""
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using basic regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for RAG retrieval."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ─── Timing ───────────────────────────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


# ─── Streamlit UI Helpers ─────────────────────────────────────────────────────

def render_badge(label: str, value: float, color: str = "#6C63FF") -> str:
    """Return an HTML badge string for confidence display."""
    pct = f"{value * 100:.1f}%"
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:20px;font-size:0.85em;font-weight:600;">'
        f"{label}: {pct}</span>"
    )


def progress_bar_html(value: float, color: str = "#6C63FF", label: str = "") -> str:
    """Return an HTML progress bar."""
    pct = value * 100
    return f"""
    <div style="margin:4px 0;">
      <div style="display:flex;justify-content:space-between;font-size:0.8em;color:#8B949E;">
        <span>{label}</span><span>{pct:.1f}%</span>
      </div>
      <div style="background:#21262D;border-radius:6px;height:8px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{color};
                    border-radius:6px;transition:width 0.4s;"></div>
      </div>
    </div>
    """


def info_card(title: str, body: str, color: str = "#6C63FF") -> str:
    """Return a styled HTML card."""
    return f"""
    <div style="border-left:4px solid {color};background:#161B22;
                padding:14px 18px;border-radius:0 8px 8px 0;margin:8px 0;">
      <p style="margin:0;font-weight:700;color:{color};font-size:0.9em;">{title}</p>
      <p style="margin:4px 0 0;color:#E6EDF3;font-size:0.95em;">{body}</p>
    </div>
    """


def metric_card(label: str, value: Any, delta: str = "", color: str = "#6C63FF") -> str:
    """Return a styled metric card."""
    delta_html = f'<span style="color:#06D6A0;font-size:0.8em;">{delta}</span>' if delta else ""
    return f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:12px;
                padding:16px 20px;text-align:center;margin:4px;">
      <p style="color:#8B949E;font-size:0.8em;margin:0;">{label}</p>
      <p style="color:{color};font-size:1.8em;font-weight:700;margin:4px 0;">{value}</p>
      {delta_html}
    </div>
    """


def styled_header(icon: str, title: str, subtitle: str, color: str = "#6C63FF"):
    """Render a styled page header in Streamlit."""
    st.markdown(f"""
    <div style="padding:20px 0 10px;">
      <h1 style="font-size:2em;font-weight:800;color:{color};margin:0;">
        {icon} {title}
      </h1>
      <p style="color:#8B949E;font-size:1em;margin:4px 0 0;">{subtitle}</p>
      <hr style="border-color:#30363D;margin-top:12px;">
    </div>
    """, unsafe_allow_html=True)


def example_button_row(examples: list[str], key_prefix: str) -> str | None:
    """Render example text buttons. Returns selected example text or None."""
    st.markdown("**💡 Try an example:**")
    cols = st.columns(min(len(examples), 3))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"Example {i+1}", key=f"{key_prefix}_ex_{i}", use_container_width=True):
                return ex
    return None


def no_model_warning(model_name: str):
    """Display a user-friendly model loading notice."""
    st.info(
        f"🔄 Model **{model_name}** will be downloaded on first use "
        f"(cached for subsequent runs). This may take a moment.",
        icon="ℹ️",
    )
