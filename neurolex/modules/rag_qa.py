"""
Module 04 — Retrieval-Augmented Question Answering (RAG)
TF-IDF retrieval of relevant passages → RoBERTa extractive QA.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from neurolex.config import MODELS
from neurolex.utils import split_into_chunks


@st.cache_resource(show_spinner=False)
def _load_qa_pipeline():
    cfg = MODELS["qa"]
    return pipeline(cfg["task"], model=cfg["model_name"])


class RAGQuestionAnswerer:
    """
    Retrieval-Augmented Generation QA System.

    Architecture:
        Retriever: TF-IDF sparse retrieval (production: DPR / BM25)
        Reader:    deepset/roberta-base-squad2 extractive QA
        Pipeline:  Document → Chunk → Retrieve top-k → QA per chunk → Aggregate

    Evaluation:
        - Metrics: Exact Match (EM), F1, ROUGE-L
        - Benchmarks: SQuAD2, Natural Questions, TriviaQA
    """

    def __init__(self):
        self.qa_pipe = _load_qa_pipeline()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.chunks: list[str] = []
        self.tfidf_matrix = None

    def index_documents(self, documents: list[str], chunk_size: int = 300) -> int:
        """
        Split documents into chunks and build TF-IDF index.

        Returns:
            Number of chunks indexed
        """
        self.chunks = []
        for doc in documents:
            self.chunks.extend(split_into_chunks(doc, chunk_size=chunk_size))
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve most relevant chunks for a query using TF-IDF cosine similarity.
        """
        if self.tfidf_matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self.chunks[i], "score": float(scores[i]), "chunk_id": int(i)}
            for i in top_idx
            if scores[i] > 0
        ]

    def answer(self, question: str, top_k: int = 3) -> dict:
        """
        Full RAG pipeline: retrieve context → extract answer.

        Returns:
            dict with answer, score, context, retrieved_chunks
        """
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "No relevant context found. Please add documents to the corpus.",
                "score": 0.0,
                "context": "",
                "retrieved_chunks": [],
            }

        # Concatenate top-k chunks as context
        context = " ".join(c["chunk"] for c in retrieved)

        try:
            result = self.qa_pipe(question=question, context=context[:2000])
            return {
                "answer": result["answer"],
                "score": round(result["score"], 4),
                "context": context[:500] + "…",
                "retrieved_chunks": retrieved,
                "start": result.get("start"),
                "end": result.get("end"),
            }
        except Exception as e:
            return {
                "answer": f"Error during QA: {str(e)}",
                "score": 0.0,
                "context": context[:300],
                "retrieved_chunks": retrieved,
            }
