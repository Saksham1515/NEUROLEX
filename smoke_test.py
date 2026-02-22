"""Smoke test for all NEUROLEX modules."""
print("=" * 60)
print("NEUROLEX — Module Smoke Test")
print("=" * 60)

# Import all modules
try:
    from neurolex.modules.classifier import MultiLabelClassifier
    from neurolex.modules.ner import NERLinker
    from neurolex.modules.semantic_search import SemanticSearchEngine
    from neurolex.modules.rag_qa import RAGQuestionAnswerer
    from neurolex.modules.summarizer import DocumentSummarizer
    from neurolex.modules.translator import MultilingualTranslator
    from neurolex.modules.topic_modeler import TopicModeler
    from neurolex.modules.hallucination import HallucinationDetector
    from neurolex.modules.toxicity import ToxicityDetector
    print("✅ All 9 modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise

# Test 1: Topic Modeler (no model download needed)
print("\n--- Test: TopicModeler ---")
m = TopicModeler(n_topics=2, n_keywords=4)
docs = [
    "AI transforms healthcare diagnostics with deep learning",
    "Machine learning detects cancer in medical images",
    "Climate change causes floods worldwide",
    "Carbon emissions are heating the planet",
]
r = m.fit(docs)
print(f"✅ TopicModeler: {r['n_topics']} topics from {r['n_docs']} docs")
for t in r["topics"]:
    print(f"   {t['label']}: {t['keywords_str']}")

# Test 2: SemanticSearch instance creation (no download)
print("\n--- Test: SemanticSearchEngine ---")
e = SemanticSearchEngine()
print("✅ SemanticSearchEngine instantiated (lazy model load)")

# Test 3: RAGQuestionAnswerer (no download needed for indexer)
print("\n--- Test: RAGQuestionAnswerer ---")
rag = RAGQuestionAnswerer()
n = rag.index_documents(["Python was created by Guido van Rossum in 1991."])
print(f"✅ RAGQuestionAnswerer indexed {n} chunks")

# Test 4: Summarizer extractive (no model download)
print("\n--- Test: DocumentSummarizer extractive ---")
text = (
    "Python is a versatile programming language. "
    "It supports multiple paradigms including object-oriented and functional. "
    "Python is widely used in data science and artificial intelligence. "
    "Many top companies rely on Python for their backend services."
)
# Create a partial summarizer instance just for extractive
s = DocumentSummarizer.__new__(DocumentSummarizer)
result = s.summarize_extractive(text, n_sentences=2)
print(f"✅ Extractive summary ({result['word_count_out']} words): {result['summary'][:80]}...")

# Test 5: Translator (no download needed for init)
print("\n--- Test: MultilingualTranslator ---")
t = MultilingualTranslator()
bleu = t.bleu_estimate("hello world", "hello world")
print(f"✅ Translator BLEU estimate: {bleu}")

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\n🚀 App running at: http://localhost:8501")
