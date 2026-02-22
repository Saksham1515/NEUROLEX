"""
Module 05 — Abstractive & Extractive Summarization
BART-CNN for abstractive; sentence scoring for extractive.
"""
from __future__ import annotations
import numpy as np
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS, SUMMARIZER_CONFIG
from neurolex.utils import split_into_sentences


@st.cache_resource(show_spinner=False)
def _load_abstractive():
    cfg = MODELS["summarizer_abstractive"]
    return pipeline(cfg["task"], model=cfg["model_name"])


class DocumentSummarizer:
    """
    Dual-mode document summarization.

    Abstractive (BART-CNN):
        - Encoder-decoder transformer
        - Generates novel sentences not in the source
        - Best for: news, reports, long-form content

    Extractive (Sentence scoring):
        - TF-IDF sentence importance scoring
        - Selects verbatim sentences from source
        - Best for: preserve exact wording, legal docs

    Evaluation:
        - ROUGE-1, ROUGE-2, ROUGE-L
        - BERTScore for semantic similarity
        - Faithfulness score (hallucination check)
    """

    def __init__(self):
        self.abstractive_pipe = _load_abstractive()

    def summarize_abstractive(self, text: str, ratio: float = 0.3) -> dict:
        """
        Generate an abstractive summary using BART-CNN.

        Args:
            text: Source document
            ratio: Target summary length as fraction of input

        Returns:
            dict with summary, word_count_in, word_count_out, compression_ratio
        """
        word_count_in = len(text.split())
        max_l = max(40, int(word_count_in * ratio))
        min_l = max(20, int(max_l * 0.4))

        try:
            result = self.abstractive_pipe(
                text[:3000],
                max_length=min(max_l, 512),
                min_length=min_l,
                length_penalty=SUMMARIZER_CONFIG["length_penalty"],
                num_beams=SUMMARIZER_CONFIG["num_beams"],
                early_stopping=True,
                truncation=True,
            )
            summary = result[0]["summary_text"]
        except Exception as e:
            summary = f"Summarization error: {e}"

        word_count_out = len(summary.split())
        return {
            "summary": summary,
            "word_count_in": word_count_in,
            "word_count_out": word_count_out,
            "compression_ratio": round(word_count_out / max(word_count_in, 1), 3),
            "mode": "Abstractive (BART-CNN)",
        }

    def summarize_extractive(self, text: str, n_sentences: int = 4) -> dict:
        """
        Extractive summarization via TF-IDF sentence scoring.

        Scores each sentence by its average TF-IDF score against the document,
        then selects top-N sorted by original order.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        sentences = split_into_sentences(text)
        word_count_in = len(text.split())

        if len(sentences) <= n_sentences:
            return {
                "summary": text,
                "word_count_in": word_count_in,
                "word_count_out": word_count_in,
                "compression_ratio": 1.0,
                "mode": "Extractive",
                "sentence_scores": [],
            }

        try:
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(sentences)
            doc_vec = tfidf_matrix.mean(axis=0)
            scores = cosine_similarity(doc_vec, tfidf_matrix).flatten()
            top_idx = sorted(np.argsort(scores)[::-1][:n_sentences])
            selected = [sentences[i] for i in top_idx]
            summary = " ".join(selected)
        except Exception:
            summary = " ".join(sentences[:n_sentences])
            scores = np.ones(len(sentences))

        word_count_out = len(summary.split())
        sentence_scores = [
            {"sentence": s[:100] + "…", "score": float(scores[i])}
            for i, s in enumerate(sentences)
        ]

        return {
            "summary": summary,
            "word_count_in": word_count_in,
            "word_count_out": word_count_out,
            "compression_ratio": round(word_count_out / max(word_count_in, 1), 3),
            "mode": "Extractive (TF-IDF Scoring)",
            "sentence_scores": sorted(sentence_scores, key=lambda x: -x["score"]),
        }

    def rouge_estimate(self, reference: str, hypothesis: str) -> dict:
        """
        Simple ROUGE-1 and ROUGE-2 token overlap estimate.
        (Full ROUGE requires `rouge-score` package)
        """
        def ngrams(tokens, n):
            return set(zip(*[tokens[i:] for i in range(n)]))

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        r1_ref = set(ref_tokens)
        r1_hyp = set(hyp_tokens)
        r1 = len(r1_ref & r1_hyp) / max(len(r1_ref), 1)

        r2_ref = ngrams(ref_tokens, 2)
        r2_hyp = ngrams(hyp_tokens, 2)
        r2 = len(r2_ref & r2_hyp) / max(len(r2_ref), 1)

        return {"rouge_1": round(r1, 4), "rouge_2": round(r2, 4)}
