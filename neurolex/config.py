"""
NEUROLEX — Central Configuration
All model names, device settings, and hyperparameters.
"""
import torch

#   Headers for Wikipedia API requests (with contact info for responsible use)
HEADERS = {"User-Agent": "NEUROLEX/1.0 (contact: your_email@example.com)"}

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Module Model Configs ─────────────────────────────────────────────────────
MODELS = {
    "classifier": {
        "model_name": "facebook/bart-large-mnli",
        "task": "zero-shot-classification",
        "description": "Multi-label zero-shot classification via BART-MNLI",
    },
    "ner": {
        "model_name": "dslim/bert-base-NER",
        "task": "ner",
        "aggregation": "simple",
        "description": "BERT fine-tuned on CoNLL-2003 for token classification",
    },
    "semantic_search": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Lightweight sentence embeddings for semantic similarity",
    },
    "qa": {
        "model_name": "deepset/roberta-base-squad2",
        "task": "question-answering",
        "description": "RoBERTa fine-tuned on SQuAD2 for extractive QA",
    },
    "summarizer_abstractive": {
        "model_name": "facebook/bart-large-cnn",
        "task": "summarization",
        "description": "BART fine-tuned on CNN/DailyMail for abstractive summarization",
    },
    "translator": {
        "model_name_template": "Helsinki-NLP/opus-mt-{src}-{tgt}",
        "task": "translation",
        "description": "Helsinki-NLP OPUS-MT multilingual translation models",
    },
    "dialogue": {
        "model_name": "microsoft/DialoGPT-medium",
        "task": "text-generation",
        "description": "DialoGPT medium — conversational language model",
    },
    "hallucination": {
        "model_name": "facebook/bart-large-mnli",
        "task": "zero-shot-classification",
        "description": "NLI-based factuality scoring using BART-MNLI",
    },
    "toxicity": {
        "model_name": "unitary/toxic-bert",
        "task": "text-classification",
        "description": "Multi-label toxicity detection (6 categories)",
    },
}

# ─── Translation Language Pairs ───────────────────────────────────────────────
TRANSLATION_PAIRS = {
    "English → French": ("en", "fr"),
    "English → German": ("en", "de"),
    "English → Spanish": ("en", "es"),
    "English → Italian": ("en", "it"),
    "English → Portuguese": ("en", "pt"),
    "English → Hindi": ("en", "hi"),
    "English → Chinese": ("en", "zh"),
    "English → Japanese": ("en", "jap"),
    "English → Arabic": ("en", "ar"),
    "French → English": ("fr", "en"),
    "German → English": ("de", "en"),
    "Spanish → English": ("es", "en"),
    "Italian → English": ("it", "en"),
    "Portuguese → English": ("pt", "en"),
}

# ─── Summarization ────────────────────────────────────────────────────────────
SUMMARIZER_CONFIG = {
    "max_length": 256,
    "min_length": 40,
    "length_penalty": 2.0,
    "num_beams": 4,
    "early_stopping": True,
}

# ─── Generation ───────────────────────────────────────────────────────────────
DIALOGUE_CONFIG = {
    "max_new_tokens": 200,
    "do_sample": True,
    "top_p": 0.95,
    "temperature": 0.75,
    "pad_token_id": 50256,
}

# ─── Topic Modeling ───────────────────────────────────────────────────────────
TOPIC_CONFIG = {
    "n_topics": 5,
    "n_keywords": 8,
    "min_docs_per_topic": 1,
}

# ─── UI Theme Colors ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#48CAE4",
    "accent": "#F72585",
    "success": "#06D6A0",
    "warning": "#FFB703",
    "danger": "#EF233C",
    "background": "#0D1117",
    "surface": "#161B22",
    "surface2": "#21262D",
    "text": "#E6EDF3",
    "muted": "#8B949E",
}

# ─── Module Metadata ──────────────────────────────────────────────────────────
MODULE_INFO = [
    {
        "icon": "🏷️",
        "name": "Text Classification",
        "page": "01_Text_Classification",
        "color": "#6C63FF",
        "desc": "Multi-label zero-shot classification",
    },
    {
        "icon": "🔍",
        "name": "NER & Entity Linking",
        "page": "02_NER_Entity_Linking",
        "color": "#48CAE4",
        "desc": "Named entity recognition + Wikipedia linking",
    },
    {
        "icon": "🔎",
        "name": "Semantic Search",
        "page": "03_Semantic_Search",
        "color": "#F72585",
        "desc": "Dense vector semantic retrieval",
    },
    {
        "icon": "💡",
        "name": "Q&A / RAG",
        "page": "04_QA_RAG",
        "color": "#06D6A0",
        "desc": "Retrieval-augmented question answering",
    },
    {
        "icon": "📝",
        "name": "Summarization",
        "page": "05_Summarization",
        "color": "#FFB703",
        "desc": "Abstractive & extractive summarization",
    },
    {
        "icon": "🌍",
        "name": "Translation",
        "page": "06_Translation",
        "color": "#4CC9F0",
        "desc": "Multilingual machine translation",
    },
    {
        "icon": "📊",
        "name": "Topic Modeling",
        "page": "07_Topic_Modeling",
        "color": "#7209B7",
        "desc": "Neural topic discovery & trend detection",
    },
    {
        "icon": "💬",
        "name": "Dialogue System",
        "page": "08_Dialogue_System",
        "color": "#F77F00",
        "desc": "Multi-turn conversational AI",
    },
    {
        "icon": "⚠️",
        "name": "Hallucination Detection",
        "page": "09_Hallucination_Detection",
        "color": "#EF233C",
        "desc": "NLI-based factuality scoring",
    },
    {
        "icon": "🛡️",
        "name": "Bias & Toxicity",
        "page": "10_Bias_Toxicity",
        "color": "#2EC4B6",
        "desc": "Multi-label toxicity & bias detection",
    },
]
