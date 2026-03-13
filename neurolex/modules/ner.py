"""
Module 02 — Named Entity Recognition + Entity Linking
BERT-NER for extraction, Wikipedia API for linking.
"""
from __future__ import annotations
import streamlit as st
from transformers import pipeline
from neurolex.config import MODELS,HEADERS
# import utils
# from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def _load_ner_pipeline():
    cfg = MODELS["ner"]
    return pipeline(
        cfg["task"],
        model=cfg["model_name"],
        aggregation_strategy=cfg["aggregation"],
    )
    
# @st.cache_resource(show_spinner=False)
# def _load_embedding_model():
#     return SentenceTransformer(MODELS["semantic_search"]["model_name"])


class NERLinker:
    """
    Named Entity Recognition (BERT-NER) + Wikipedia Entity Linking.

    Architecture:
        - NER: dslim/bert-base-NER (CoNLL-2003)
        - Entity types: PER, ORG, LOC, MISC
        - Linking: Wikipedia REST API summary endpoint
    """

    ENTITY_COLORS = {
        "PER": "#6C63FF",
        "ORG": "#F72585",
        "LOC": "#06D6A0",
        "MISC": "#FFB703",
    }

    def __init__(self):
        self.ner = _load_ner_pipeline()
        # self.embedder = _load_embedding_model()

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extract named entities from text.

        Returns:
            List of dicts with: entity_group, word, score, start, end
        """
        if not text.strip():
            return []
        results = self.ner(text)
        return [
            {
                "entity": r["entity_group"],
                "word": r["word"],
                "score": round(r["score"], 4),
                "start": r["start"],
                "end": r["end"],
                "color": self.ENTITY_COLORS.get(r["entity_group"], "#8B949E"),
            }
            for r in results
        ]
    
    # This method is an alternative approach to entity linking using semantic search with sentence transformers.
    # !!!(doesn't perform well in practice for short entity names, but can be useful for longer phrases or ambiguous cases)
    
    # def _search_wikipedia(self,user_input: str, entity: str, limit: int = 3):
    #     url = "https://en.wikipedia.org/w/api.php"

    #     params = {
    #         "action": "query",
    #         "list": "search",
    #         "srsearch": entity,
    #         "format": "json",
    #         "srlimit": limit,
    #     }
    #     result = {}
    #     try:
    #         r = requests.get(url, params=params, headers=HEADERS, timeout=8)
    #         data = r.json()
    #         titles = [item["title"] for item in data.get("query", {}).get("search", [])]
    #         for each_title in titles:
    #             snippit_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{each_title.replace(' ', '_')}"
    #             r = requests.get(snippit_url, headers=HEADERS, timeout=8)
    #             data = r.json()
    #             result[data.get("title", each_title)] = data.get("extract", "No description found.")[:400]
            
    #         embedder = self.embedder
    #         context_vec = self.embedder.encode(user_input, convert_to_tensor=True)
    #         candidate_vecs = embedder.encode(list(result.values()), convert_to_tensor=True)
    #         scores = utils.cos_sim(context_vec, candidate_vecs)
    #         best_idx = scores.argmax().item()
    #         best_title = list(result.keys())[best_idx]
    #         return best_title
    #     except Exception:
    #         return entity
        
        

    def link_entity(self, entity_name: str) -> dict:
        """
        Link entity to Wikipedia using summary API.

        Returns:
            dict with title, summary, url, thumbnail
        """
        import requests

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity_name.replace(' ', '_')}"
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "title": data.get("title", entity_name),
                    "summary": data.get("extract", "No description found.")[:400],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "thumbnail": data.get("thumbnail", {}).get("source", ""),
                    "found": True,
                }
        except Exception:
            pass
        return {"title": entity_name, "summary": "Entity not found in Wikipedia.", "found": False}

    def annotate_html(self, text: str, entities: list[dict]) -> str:
        """
        Return HTML with colored entity spans highlighted inline.
        """
        highlighted = text
        # Sort by start position descending to avoid offset issues
        for ent in sorted(entities, key=lambda x: -x["start"]):
            etype = ent["entity"]
            escore = ent["score"]
            ecolor = ent["color"]
            eword = ent["word"]
            title_attr = f"{etype}: {escore:.2%}"
            span = (
                f'<mark style="background:{ecolor}33;color:{ecolor};'
                f'border-radius:4px;padding:1px 4px;font-weight:600;" '
                f'title="{title_attr}">'
                f'{eword}<sup style="font-size:0.6em">{etype}</sup></mark>'
            )
            highlighted = highlighted[: ent["start"]] + span + highlighted[ent["end"] :]
        return f'<p style="line-height:2;font-size:1em;">{highlighted}</p>'

    def get_entity_stats(self, entities: list[dict]) -> dict:
        """Aggregate entity counts by type."""
        from collections import Counter
        counts = Counter(e["entity"] for e in entities)
        return dict(counts)
