import re
import unicodedata
from pathlib import Path
from typing import List


class SimpleRAG:
    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.doc_texts: List[str] = []
        self.doc_names: List[str] = []
        self.doc_tokens: List[set[str]] = []
        self.reindex()

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        normalized = unicodedata.normalize("NFKD", lowered)
        no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return no_accents

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        return set(re.findall(r"[\w\u0600-\u06FF]+", cls._normalize_text(text)))

    @staticmethod
    def _extract_snippet(text: str, query_tokens: set[str], max_len: int = 650) -> str:
        if not text:
            return ""

        lowered = text.lower()
        hit_index = -1
        for tok in sorted(query_tokens, key=len, reverse=True):
            idx = lowered.find(tok)
            if idx >= 0:
                hit_index = idx
                break

        if hit_index < 0:
            return text[:max_len].replace("\n", " ")

        start = max(0, hit_index - max_len // 3)
        end = min(len(text), start + max_len)
        snippet = text[start:end].replace("\n", " ")
        return snippet

    def reindex(self) -> None:
        files = sorted(self.docs_dir.glob("*.md"))
        self.doc_names = [f.name for f in files]
        self.doc_texts = [f.read_text(encoding="utf-8") for f in files]
        self.doc_tokens = [self._tokenize(text) for text in self.doc_texts]

    def retrieve(self, query: str, top_k: int = 3):
        if not self.doc_texts or not query.strip():
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        ranked_scores: list[tuple[int, float]] = []
        for idx, d_tokens in enumerate(self.doc_tokens):
            if not d_tokens:
                ranked_scores.append((idx, 0.0))
                continue
            overlap = q_tokens.intersection(d_tokens)
            overlap_ratio = len(overlap) / max(len(q_tokens), 1)
            doc_coverage = len(overlap) / max(len(d_tokens), 1)
            score = 0.85 * overlap_ratio + 0.15 * min(doc_coverage * 10.0, 1.0)
            ranked_scores.append((idx, float(score)))

        ranked = sorted(ranked_scores, key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            snippet = self._extract_snippet(self.doc_texts[idx], q_tokens, max_len=650)
            results.append({"source": self.doc_names[idx], "score": float(score), "snippet": snippet})
        return results
