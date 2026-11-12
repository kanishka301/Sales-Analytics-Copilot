"""Retrieve relevant schema notes and SQL examples using lightweight TF-IDF search."""
import json
import re
from functools import lru_cache

from src.config import KNOWLEDGE_DIR


def _load_chunks():
    chunks = []
    dictionary = KNOWLEDGE_DIR / "data_dictionary.md"
    examples = KNOWLEDGE_DIR / "sql_examples.json"

    if dictionary.exists():
        for block in re.split(r"\n\s*\n", dictionary.read_text(encoding="utf-8")):
            if block.strip():
                chunks.append(block.strip())

    if examples.exists():
        for item in json.loads(examples.read_text(encoding="utf-8")):
            chunks.append(f"Example question: {item['question']}\nSQL: {item['sql']}")

    if not chunks:
        raise FileNotFoundError(f"No schema knowledge files found in {KNOWLEDGE_DIR}")
    return chunks


@lru_cache(maxsize=1)
def _search_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    chunks = _load_chunks()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def lookup_schema(question: str, k: int = 4) -> dict:
    """Return the most relevant data-dictionary notes and example SQL."""
    from sklearn.metrics.pairwise import cosine_similarity

    chunks, vectorizer, matrix = _search_index()
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = scores.argsort()[::-1][:max(1, int(k))]
    matches = [chunks[i] for i in best if scores[i] > 0]
    return {"retriever": "tfidf", "context": matches}
