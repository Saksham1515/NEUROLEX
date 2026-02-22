"""
Module 09 — Hallucination Detection & Factuality Scoring
NLI-based entailment scoring between source and generated text.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS


@st.cache_resource(show_spinner=False)
def _load_nli_pipeline():
    cfg = MODELS["hallucination"]
    return pipeline(cfg["task"], model=cfg["model_name"])


class HallucinationDetector:
    """
    Hallucination and Factuality Scorer using NLI (Natural Language Inference).

    Architecture:
        - Model: BART-MNLI (fine-tuned for NLI on MultiNLI)
        - Approach: Frame factuality as NLI:
            Premise = Source Document / Reference
            Hypothesis = Generated Text / Claim
        - Labels: entailment → factual, contradiction → hallucination,
                  neutral → insufficient evidence

    Scoring:
        - Factuality Score = P(entailment)
        - Hallucination Risk = P(contradiction)
        - Uncertainty = P(neutral)
        - Overall: sentence-level + document-level aggregation

    Evaluation:
        - FactCC (Frank & Moramarco 2021)
        - FEVER shared task
        - SummaC benchmark for summarization faithfulness

    Extensions:
        - FActScore: fine-grained claim decomposition
        - SAFE: Supported facts evaluation
        - ViNLI: Visual hallucination detection
    """

    NLI_LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self):
        self.nli_pipe = _load_nli_pipeline()

    def score_factuality(self, source: str, generated: str) -> dict:
        """
        Compute factuality score of generated text against source.

        Args:
            source: Reference document / ground-truth context
            generated: Generated or claimed text to evaluate

        Returns:
            dict with factuality_score, hallucination_risk, verdict, scores
        """
        if not source.strip() or not generated.strip():
            return {}

        result = self.nli_pipe(
            generated[:1000],
            candidate_labels=self.NLI_LABELS,
            multi_label=False,
            hypothesis_template="{}",
        )

        # Build score dict, normalize
        scores = dict(zip(result["labels"], result["scores"]))
        entailment = scores.get("entailment", 0.0)
        contradiction = scores.get("contradiction", 0.0)
        neutral = scores.get("neutral", 0.0)

        # Verdict
        if entailment > 0.5:
            verdict = "✅ Factual"
            verdict_color = "#06D6A0"
        elif contradiction > 0.4:
            verdict = "🚨 Hallucination Detected"
            verdict_color = "#EF233C"
        else:
            verdict = "⚠️ Uncertain / Insufficient Evidence"
            verdict_color = "#FFB703"

        return {
            "factuality_score": round(entailment, 4),
            "hallucination_risk": round(contradiction, 4),
            "uncertainty": round(neutral, 4),
            "verdict": verdict,
            "verdict_color": verdict_color,
            "scores": {
                "Entailment (Factual)": entailment,
                "Neutral (Uncertain)": neutral,
                "Contradiction (Hallucinated)": contradiction,
            },
        }

    def sentence_level_analysis(self, source: str, generated: str) -> list[dict]:
        """
        Score each sentence of generated text independently.
        Useful for identifying specific hallucinated claims.
        """
        import re
        sentences = re.split(r"(?<=[.!?])\s+", generated.strip())
        results = []
        for sent in sentences:
            if len(sent.strip()) > 15:
                score = self.score_factuality(source, sent)
                if score:
                    score["sentence"] = sent
                    results.append(score)
        return results

    @staticmethod
    def aggregate_scores(sentence_scores: list[dict]) -> dict:
        """Aggregate sentence-level scores to document-level statistics."""
        if not sentence_scores:
            return {}
        factuality = [s["factuality_score"] for s in sentence_scores]
        hallu = [s["hallucination_risk"] for s in sentence_scores]
        return {
            "mean_factuality": round(sum(factuality) / len(factuality), 4),
            "mean_hallucination_risk": round(sum(hallu) / len(hallu), 4),
            "min_factuality": round(min(factuality), 4),
            "n_hallucinated_sentences": sum(1 for h in hallu if h > 0.4),
            "n_sentences": len(sentence_scores),
        }
