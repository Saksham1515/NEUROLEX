"""
Module 01 — Multi-Label Text Classifier
Uses BART-MNLI zero-shot classification pipeline.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS, COLORS


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    cfg = MODELS["classifier"]
    return pipeline(cfg["task"], model=cfg["model_name"])


class MultiLabelClassifier:
    """
    Zero-shot multi-label text classifier using BART-MNLI.

    Architecture:
        - Model: facebook/bart-large-mnli
        - Approach: Natural Language Inference (NLI) framing
        - Each label is treated as a hypothesis; the model scores entailment
    """

    def __init__(self):
        self.pipe = _load_pipeline()

    def classify(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.3,
        multi_label: bool = True,
    ) -> dict:
        """
        Classify text against candidate labels.

        Args:
            text: Input text to classify
            labels: List of candidate category labels
            threshold: Minimum confidence to include a label (0–1)
            multi_label: If True, labels are scored independently

        Returns:
            dict with 'results' (label→score), 'top_label', 'above_threshold'
        """
        if not text.strip() or not labels:
            return {}

        output = self.pipe(text, candidate_labels=labels, multi_label=multi_label)
        scores = dict(zip(output["labels"], output["scores"]))

        above = {k: v for k, v in scores.items() if v >= threshold}
        top = output["labels"][0] if output["labels"] else ""

        return {
            "results": scores,
            "above_threshold": above,
            "top_label": top,
            "top_score": output["scores"][0] if output["scores"] else 0.0,
        }

    # ── Interpretability ──────────────────────────────────────────────────────

    def explain(self, text: str, labels: list[str]) -> dict:
        """
        Simple attribution: re-run classification on each sentence to show
        which part of the text drives each label's score.
        """
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        attributions = {}
        for label in labels:
            sent_scores = []
            for sent in sentences:
                if len(sent) > 10:
                    out = self.pipe(sent, candidate_labels=[label], multi_label=True)
                    sent_scores.append((sent, out["scores"][0]))
            attributions[label] = sorted(sent_scores, key=lambda x: -x[1])
        return attributions

    # ── Active Learning hint ──────────────────────────────────────────────────

    @staticmethod
    def uncertainty_score(scores: dict[str, float]) -> float:
        """
        Least-confidence uncertainty: 1 - max_score.
        High values indicate good candidates for active labeling.
        """
        if not scores:
            return 0.0
        return 1.0 - max(scores.values())
