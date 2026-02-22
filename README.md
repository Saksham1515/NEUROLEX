# 🧠 NEUROLEX — End-to-End Advanced NLP Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-grade, modular NLP research platform** covering 10 major NLP tasks, built with Python, Streamlit, and HuggingFace Transformers. Managed with `uv`.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    🧠 NEUROLEX v1.0                      │
│                  Streamlit Frontend UI                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │   Sidebar Navigation (10 NLP Modules + Home)       │  │
│  └─────────────────┬──────────────────────────────────┘  │
│                    │                                      │
│  ┌─────────────┬───┴─────────────┬──────────────────┐    │
│  │ config.py   │  Module Registry│   utils.py       │    │
│  └─────────────┴─────────────────┴──────────────────┘    │
│                                                           │
│  ┌─────────────── NLP Module Layer ───────────────────┐  │
│  │  01 Classifier  │ 02 NER+Link  │ 03 SemanticSearch │  │
│  │  04 RAG Q&A     │ 05 Summarizer│ 06 Translator     │  │
│  │  07 TopicModeler│ 08 Dialogue  │ 09 Hallucination  │  │
│  │  10 Toxicity                                        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │       HuggingFace Transformers / Pipeline API      │  │
│  │  BART · BERT · RoBERTa · DialoGPT · Helsinki-NLP  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Selection Table

| # | Task | Model | Approach | Benchmark |
|---|------|-------|----------|-----------|
| 01 | **Text Classification** | `facebook/bart-large-mnli` | Zero-shot NLI | MNLI 90.7% |
| 02 | **NER** | `dslim/bert-base-NER` | Token classification | CoNLL-03 F1: 91.3 |
| 03 | **Entity Linking** | Wikipedia REST API | API + matching | — |
| 04 | **Semantic Search** | `all-MiniLM-L6-v2` | Dense embeddings | BEIR / MS MARCO |
| 05 | **Q&A / RAG** | `deepset/roberta-base-squad2` | Extractive QA + TF-IDF | SQuAD2 F1: 84.2 |
| 06 | **Summarization** | `facebook/bart-large-cnn` | Seq2Seq generation | CNN/DM ROUGE-L: 40.9 |
| 07 | **Translation** | `Helsinki-NLP/opus-mt-*` | MarianMT | WMT BLEU varies |
| 08 | **Topic Modeling** | TF-IDF + SVD + KMeans | LSA + clustering | Topic Coherence CV |
| 09 | **Dialogue** | `microsoft/DialoGPT-medium` | Causal LM | DailyDialog PPL |
| 10 | **Hallucination** | `facebook/bart-large-mnli` | NLI entailment | FactCC / FEVER |
| 11 | **Toxicity** | `unitary/toxic-bert` | Multi-label BERT | Jigsaw AUC: 0.98 |

---

## 📂 Project Structure

```
NEUROLEX/
├── app.py                          # Main Streamlit entry point (home/architecture)
├── pyproject.toml                  # uv project config + dependencies
├── README.md
│
├── neurolex/                       # Core package
│   ├── __init__.py
│   ├── config.py                   # Model names, hyperparams, UI colors
│   ├── utils.py                    # Text utilities, Streamlit helpers
│   └── modules/                    # NLP module classes
│       ├── __init__.py
│       ├── classifier.py           # Multi-label zero-shot classification
│       ├── ner.py                  # NER + Wikipedia entity linking
│       ├── semantic_search.py      # Dense semantic retrieval
│       ├── rag_qa.py               # RAG question answering
│       ├── summarizer.py           # Abstractive + extractive summarization
│       ├── translator.py           # Multilingual machine translation
│       ├── topic_modeler.py        # Topic modeling + trend detection
│       ├── dialogue.py             # Multi-turn conversational dialogue
│       ├── hallucination.py        # NLI-based factuality scoring
│       └── toxicity.py             # Multi-label toxicity detection
│
└── pages/                          # Streamlit multi-page app pages
    ├── 01_Text_Classification.py
    ├── 02_NER_Entity_Linking.py
    ├── 03_Semantic_Search.py
    ├── 04_QA_RAG.py
    ├── 05_Summarization.py
    ├── 06_Translation.py
    ├── 07_Topic_Modeling.py
    ├── 08_Dialogue_System.py
    ├── 09_Hallucination_Detection.py
    └── 10_Bias_Toxicity.py
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone / enter project directory
cd NEUROLEX

# Install all dependencies (uv creates venv automatically)
uv sync

# Run the Streamlit app
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Training & Fine-Tuning Pipeline

### 1. Data Preparation
```
Raw corpus → Language detection → Normalization → Deduplication
→ Task annotation → Train/Val/Test split (80/10/10) → DataLoader
```

### 2. Fine-Tuning Strategy
| Strategy | Use Case |
|----------|----------|
| Full fine-tune | Classification, NER (smaller models) |
| PEFT / LoRA | Large LMs (BART, GPT) — 10x fewer params |
| Adapter layers | Domain adaptation without full retraining |
| Continued pre-training | Domain-specific corpora (medical, legal) |

### 3. Optimizer Settings
- **Optimizer:** AdamW (weight decay 0.01)
- **Scheduler:** Linear warmup (10%) → cosine decay
- **Gradient clipping:** max_norm = 1.0
- **Mixed precision:** fp16 / bf16

### 4. Active Learning (Query Strategy)
- **Least confidence:** select samples where `1 - max_score` is highest
- **Entropy sampling:** `H(y|x) = -Σ p log p`
- **Core-set selection:** maximize coverage in embedding space

### 5. Continual Learning
- **Elastic Weight Consolidation (EWC):** penalize changes to important weights
- **Experience Replay:** maintain memory buffer of past task examples
- **Progressive Neural Networks:** add new columns for new tasks

---

## 📊 Evaluation Framework

| Task | Primary Metric | Secondary | Dataset |
|------|---------------|-----------|---------|
| Classification | Macro-F1, AUC-ROC | Precision, Recall | MultiLabel NewsGroups |
| NER | Span F1 | Entity-level precision | CoNLL-2003 |
| Semantic Search | NDCG@10, MRR | Recall@K | MS MARCO / BEIR |
| Q&A | Exact Match, F1 | ROUGE-L | SQuAD 2.0 |
| Summarization | ROUGE-1/2/L, BERTScore | Faithfulness | CNN/DM, XSum |
| Translation | BLEU, chrF++ | TER | WMT14/19 |
| Topic Modeling | Coherence CV, Diversity | Perplexity | 20 Newsgroups |
| Dialogue | Perplexity, BLEU-4 | Human eval | DailyDialog |
| Hallucination | FactCC, ViNLI | NLI accuracy | FEVER |
| Toxicity | AUC-ROC per label, F1 | Fairness metrics | Jigsaw |

---

## 🔬 Explainability & Interpretability

| Method | Applicable To | Library |
|--------|--------------|---------|
| LIME | All classifiers | `lime` |
| SHAP | All classifiers | `shap` |
| Attention Rollout | BERT-based models | `bertviz` |
| Integrated Gradients | Any differentiable model | `captum` |
| Probing Tasks | Encoder representations | Custom |
| Model Cards | All modules | `model-card-toolkit` |

---

## 🌐 API Design (FastAPI Extension)

```python
# Proposed REST API endpoints
POST /api/v1/classify       # Multi-label classification
POST /api/v1/ner            # Named entity recognition
POST /api/v1/search         # Semantic search
POST /api/v1/qa             # Question answering
POST /api/v1/summarize      # Document summarization
POST /api/v1/translate      # Machine translation
POST /api/v1/topics         # Topic modeling
POST /api/v1/chat           # Conversational dialogue
POST /api/v1/factuality     # Hallucination detection
POST /api/v1/toxicity       # Toxicity classification
GET  /api/v1/health         # Health check
GET  /api/v1/models         # Model registry info
```

---

## 🚀 Production Deployment

```
Docker (multi-stage build)
    → FastAPI backend + Streamlit frontend
    → Nginx reverse proxy + HTTPS/TLS
    → Kubernetes / Docker Compose orchestration

Model Serving:
    → ONNX export for CPU-optimized inference
    → TorchServe / Triton Inference Server
    → vLLM for large generative models

Monitoring:
    → Prometheus + Grafana (latency, throughput)
    → Evidently AI (data drift, model decay)
    → Sentry (error tracking)
```

---

## 🔭 Future Research Extensions

1. **Retrieval-Augmented Generation (RAG 2.0)** — DPR + FAISS + chunk re-ranking
2. **Vision-Language Models** — CLIP/LLaVA for multimodal NLP
3. **Few-Shot & Zero-Shot via Instruction Tuning** — FLAN-T5, Alpaca-style
4. **Cross-Lingual Transfer** — mBERT, XLM-R for low-resource languages
5. **Constitutional AI & RLHF** — Human preference alignment
6. **Federated Learning** — Privacy-preserving distributed NLP
7. **Neurosymbolic NLP** — Combining neural models with logic rules
8. **Efficient Attention** — Longformer, BigBird for 100K+ token documents

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
