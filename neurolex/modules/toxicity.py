"""
Module 10 — Bias & Toxicity Detection
ToxicBERT multi-label toxicity classification.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS


@st.cache_resource(show_spinner=False)
def _load_toxicity_pipeline():
    cfg = MODELS["toxicity"]
    return pipeline(cfg["task"], model=cfg["model_name"], top_k=None)


# Toxicity category descriptions
CATEGORY_INFO = {
    "toxic": {
        "label": "Toxic",
        "desc": "General toxicity / hateful content",
        "color": "#EF233C",
        "icon": "☠️",
    },
    "severe_toxic": {
        "label": "Severely Toxic",
        "desc": "Extreme hatred or violent threats",
        "color": "#B5000E",
        "icon": "💀",
    },
    "obscene": {
        "label": "Obscene",
        "desc": "Explicit or vulgar language",
        "color": "#F77F00",
        "icon": "🤬",
    },
    "threat": {
        "label": "Threat",
        "desc": "Direct threats of harm or violence",
        "color": "#7B0D1E",
        "icon": "⚠️",
    },
    "insult": {
        "label": "Insult",
        "desc": "Personal attacks and derogatory language",
        "color": "#C77DFF",
        "icon": "🗯️",
    },
    "identity_hate": {
        "label": "Identity Hate",
        "desc": "Hate speech targeting identity groups",
        "color": "#FF6B6B",
        "icon": "🎯",
    },
}

SAFE_THRESHOLD = 0.15
WARN_THRESHOLD = 0.40


class ToxicityDetector:
    """
    Multi-label toxicity and bias detection using ToxicBERT.

    Architecture:
        - Model: unitary/toxic-bert (BERT fine-tuned on Jigsaw Toxic Comment dataset)
        - Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
        - Output: Per-label probability scores (not mutually exclusive)

    Dataset:
        - Jigsaw Toxic Comment Classification Challenge (Kaggle)
        - 160K Wikipedia comments (human-labeled)

    Evaluation:
        - AUC-ROC per category
        - F1 at threshold 0.5
        - Fairness metrics: equal opportunity across demographic groups

    Bias Considerations:
        - Model trained on Wikipedia comments may miss Twitter/forum-specific slang
        - Identity term co-occurrence bias: model may flag identity mentions unfairly
        - Mitigation: calibrated thresholds per category, human review queue

    Extensions:
        - Perspective API integration
        - Cross-lingual toxicity (mBERT-based)
        - Counterfactual data augmentation for debiasing
    """

    def __init__(self):
        self.pipe = _load_toxicity_pipeline()

    def detect(self, text: str) -> dict:
        """
        Run multi-label toxicity detection on input text.

        Returns:
            dict with scores per category, overall_verdict, risk_level
        """
        if not text.strip():
            return {}

        raw = self.pipe(text[:512])
        # Flatten results (pipeline returns list of list)
        if isinstance(raw[0], list):
            raw = raw[0]

        scores = {}
        for item in raw:
            label = item["label"].lower().replace("-", "_")
            scores[label] = round(item["score"], 4)

        # Determine overall verdict
        max_score = max(scores.values()) if scores else 0.0
        flagged = {k: v for k, v in scores.items() if v >= WARN_THRESHOLD}

        if max_score < SAFE_THRESHOLD:
            verdict = "✅ Safe"
            risk_level = "None"
            risk_color = "#06D6A0"
        elif max_score < WARN_THRESHOLD:
            verdict = "⚠️ Low Risk"
            risk_level = "Low"
            risk_color = "#FFB703"
        elif max_score < 0.65:
            verdict = "🚨 Moderate Risk"
            risk_level = "Moderate"
            risk_color = "#F77F00"
        else:
            verdict = "🛑 High Risk — Content Flagged"
            risk_level = "High"
            risk_color = "#EF233C"

        return {
            "scores": scores,
            "flagged_categories": flagged,
            "overall_verdict": verdict,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "max_score": max_score,
            "category_info": CATEGORY_INFO,
        }

    def batch_detect(self, texts: list[str]) -> list[dict]:
        """Detect toxicity in a list of texts."""
        return [self.detect(t) for t in texts]

    @staticmethod
    def safe_threshold_analysis(scores: dict[str, float]) -> dict:
        """
        Return per-category safety assessment.
        """
        return {
            cat: {
                "score": score,
                "safe": score < SAFE_THRESHOLD,
                "warning": SAFE_THRESHOLD <= score < WARN_THRESHOLD,
                "flagged": score >= WARN_THRESHOLD,
            }
            for cat, score in scores.items()
        }
