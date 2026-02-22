"""
Module 07 — Topic Modeling & Trend Detection
TF-IDF + clustering for topic discovery; trend score over time.
"""
from __future__ import annotations
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from neurolex.config import TOPIC_CONFIG


class TopicModeler:
    """
    Neural-inspired topic modeling via TF-IDF + SVD + K-Means clustering.

    Architecture (lightweight BERTopic-style):
        1. TF-IDF vectorization of document collection
        2. Dimensionality reduction via TruncatedSVD (Latent Semantic Analysis)
        3. K-Means clustering to assign documents to topics
        4. Top keyword extraction per cluster for interpretability

    Production extension:
        - Replace TF-IDF with sentence-transformers embeddings
        - Replace K-Means with HDBSCAN (handles noise and variable cluster sizes)
        - Use c-TF-IDF for topic keyword extraction (BERTopic approach)

    Evaluation:
        - Topic Coherence (CV, NPMI)
        - Topic Diversity
        - Perplexity (for LDA baseline)
    """

    def __init__(self, n_topics: int = 5, n_keywords: int = 8):
        self.n_topics = n_topics
        self.n_keywords = n_keywords
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.svd = TruncatedSVD(n_components=min(50, n_topics * 4), random_state=42)
        self.kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        self.fitted = False
        self.topics: list[dict] = []
        self.document_topics: list[int] = []

    def fit(self, documents: list[str]) -> dict:
        """
        Fit topic model on a collection of documents.

        Returns:
            dict with topics (keywords + counts), coherence estimate
        """
        if len(documents) < self.n_topics:
            self.n_topics = max(1, len(documents))
            self.kmeans = KMeans(n_clusters=self.n_topics, random_state=42, n_init=10)

        tfidf_matrix = self.vectorizer.fit_transform(documents)
        vocab = np.array(self.vectorizer.get_feature_names_out())

        n_components = min(self.svd.n_components, min(tfidf_matrix.shape) - 1)
        self.svd.n_components = n_components
        reduced = self.svd.fit_transform(tfidf_matrix)
        reduced_norm = normalize(reduced)

        self.kmeans.fit(reduced_norm)
        self.document_topics = self.kmeans.labels_.tolist()

        # Extract keywords per topic using cluster centroid projections
        centers_original = self.svd.inverse_transform(self.kmeans.cluster_centers_)
        self.topics = []
        for i, center in enumerate(centers_original):
            top_word_idx = np.argsort(center)[::-1][: self.n_keywords * 2]
            keywords = [
                w for w in vocab[top_word_idx]
                if len(w) > 2 and "_" not in w
            ][: self.n_keywords]
            doc_count = int(np.sum(np.array(self.document_topics) == i))
            self.topics.append({
                "topic_id": i,
                "label": f"Topic {i + 1}",
                "keywords": keywords,
                "doc_count": doc_count,
                "keywords_str": " · ".join(keywords),
            })

        self.fitted = True
        return {
            "topics": self.topics,
            "n_docs": len(documents),
            "n_topics": self.n_topics,
            "document_assignments": self.document_topics,
        }

    def get_document_topic(self, doc_index: int) -> dict | None:
        """Return the assigned topic for a given document index."""
        if not self.fitted or doc_index >= len(self.document_topics):
            return None
        tid = self.document_topics[doc_index]
        return self.topics[tid]

    def trend_scores(self) -> list[dict]:
        """
        Return topic prevalence as trend data.
        In production: compute topic distribution per time window.
        """
        if not self.fitted:
            return []
        return sorted(self.topics, key=lambda x: -x["doc_count"])

    def infer_topic(self, new_doc: str) -> dict | None:
        """Infer the most probable topic for a new document (out-of-vocabulary)."""
        if not self.fitted:
            return None
        vec = self.vectorizer.transform([new_doc])
        reduced = self.svd.transform(vec)
        reduced_norm = normalize(reduced)
        label = int(self.kmeans.predict(reduced_norm)[0])
        return self.topics[label]
