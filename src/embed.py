import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import NOVELTY_LAMBDA, SBERT_MODEL, TOP_N_DEFAULT

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading SBERT model...")
        _model = SentenceTransformer(SBERT_MODEL)
    return _model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def encode(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def rank(
    papers: list[dict],
    interests: str,
    already_read: list[str] | None = None,
    top_n: int = TOP_N_DEFAULT,
) -> list[dict]:
    """
    Score and sort papers by novelty-adjusted relevance.
    Each paper gets two scores attached:
      relevance_score  = cosine(paper_emb, interests_emb)
      final_score      = relevance - lambda * max_sim_to_already_read
    """
    abstracts = [f"{p['title']}. {p.get('abstract', '')}" for p in papers]
    paper_embs = encode(abstracts)
    interest_emb = encode([interests])[0]

    read_embs = list(encode(already_read)) if already_read else []

    scored = []
    for paper, emb in zip(papers, paper_embs):
        relevance = _cosine(emb, interest_emb)
        if read_embs:
            max_redundancy = max(_cosine(emb, r) for r in read_embs)
        else:
            max_redundancy = 0.0
        final = relevance - NOVELTY_LAMBDA * max_redundancy
        scored.append({**paper, "relevance_score": round(relevance, 4), "final_score": round(final, 4)})

    scored.sort(key=lambda p: p["final_score"], reverse=True)
    return scored[:top_n]


def classify_tiers(papers: list[dict]) -> list[dict]:
    """
    Bucket papers into Foundational / Core / Cutting Edge
    by citation count percentile within the result set.
    Top 25% -> Foundational, bottom 25% -> Cutting Edge, rest -> Core.
    """
    counts = [p.get("citationCount") or 0 for p in papers]
    p75 = np.percentile(counts, 75)
    p25 = np.percentile(counts, 25)

    for paper in papers:
        c = paper.get("citationCount") or 0
        if p75 == p25:
            paper["tier"] = "Core"
        elif c >= p75:
            paper["tier"] = "Foundational"
        elif c <= p25:
            paper["tier"] = "Cutting Edge"
        else:
            paper["tier"] = "Core"

    return papers
