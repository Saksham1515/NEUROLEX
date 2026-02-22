"""
Module 06 — Multilingual Machine Translation
Helsinki-NLP OPUS-MT with 14 language pair options.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS, TRANSLATION_PAIRS


# Cache loaded pipelines by language pair
_translation_cache: dict[str, any] = {}


def _get_pipeline(src: str, tgt: str):
    key = f"{src}-{tgt}"
    if key not in _translation_cache:
        model_name = MODELS["translator"]["model_name_template"].format(src=src, tgt=tgt)
        try:
            _translation_cache[key] = pipeline("translation", model=model_name)
        except Exception as e:
            _translation_cache[key] = None
    return _translation_cache[key]


class MultilingualTranslator:
    """
    Multilingual machine translation using Helsinki-NLP OPUS-MT.

    Architecture:
        - Separate encoder-decoder model per language pair
        - MarianMT architecture (Vaswani et al. Transformer)
        - Trained on OPUS parallel corpora (100M+ sentence pairs)

    Evaluation:
        - BLEU score (sacrebleu)
        - chrF++ for morphologically rich languages
        - Human evaluations: adequacy + fluency

    Continual Learning:
        - Domain adaptation via continued pre-training on in-domain data
        - Language pair expansion via zero-shot multilingual models (mBART, M2M100)
    """

    LANGUAGE_PAIRS = TRANSLATION_PAIRS

    def translate(
        self,
        text: str,
        pair_name: str,
        max_length: int = 512,
    ) -> dict:
        """
        Translate text for a given language pair name.

        Args:
            text: Source text
            pair_name: Human-readable pair (e.g., 'English → French')
            max_length: Maximum output token length

        Returns:
            dict with translation, pair, source_lang, target_lang, char_count
        """
        if pair_name not in self.LANGUAGE_PAIRS:
            return {"error": f"Unknown language pair: {pair_name}"}

        src, tgt = self.LANGUAGE_PAIRS[pair_name]
        pipe = _get_pipeline(src, tgt)

        if pipe is None:
            return {
                "error": f"Model for {pair_name} could not be loaded. "
                         f"Try a different language pair."
            }

        try:
            result = pipe(text, max_length=max_length, truncation=True)
            translation = result[0].get("translation_text", "")
            return {
                "translation": translation,
                "pair": pair_name,
                "source_lang": src,
                "target_lang": tgt,
                "source_chars": len(text),
                "output_chars": len(translation),
            }
        except Exception as e:
            return {"error": str(e)}

    def batch_translate(
        self, texts: list[str], pair_name: str
    ) -> list[dict]:
        """Translate a batch of texts for a single language pair."""
        return [self.translate(t, pair_name) for t in texts]

    @staticmethod
    def bleu_estimate(reference: str, hypothesis: str) -> float:
        """
        Simplified BLEU-1 estimate using token overlap.
        Use sacrebleu for production-grade BLEU.
        """
        ref_tokens = set(reference.lower().split())
        hyp_tokens = hypothesis.lower().split()
        if not hyp_tokens:
            return 0.0
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        return round(matches / len(hyp_tokens), 4)
