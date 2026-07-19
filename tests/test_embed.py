import numpy as np
from src.embed import classify_tiers, rank

PAPERS = [
    {
        "paperId": f"id{i}",
        "title": f"Paper {i}",
        "abstract": "self-supervised contrastive learning visual representation",
        "year": 2020 + i,
        "citationCount": i * 100,
        "authors": [{"name": "Author"}],
        "venue": "NeurIPS",
    }
    for i in range(1, 6)
]


def test_rank_returns_top_n():
    ranked = rank(PAPERS, interests="contrastive learning", top_n=3)
    assert len(ranked) == 3


def test_rank_attaches_scores():
    ranked = rank(PAPERS, interests="contrastive learning", top_n=5)
    for p in ranked:
        assert "relevance_score" in p
        assert "final_score" in p
        assert 0.0 <= p["relevance_score"] <= 1.0


def test_rank_novelty_lowers_score():
    baseline = rank(PAPERS, interests="contrastive learning", top_n=5)
    penalized = rank(
        PAPERS,
        interests="contrastive learning",
        already_read=["self-supervised contrastive learning visual representation"],
        top_n=5,
    )
    avg_baseline = sum(p["final_score"] for p in baseline) / len(baseline)
    avg_penalized = sum(p["final_score"] for p in penalized) / len(penalized)
    assert avg_penalized < avg_baseline


def test_classify_tiers_assigns_all():
    ranked = rank(PAPERS, interests="contrastive learning", top_n=5)
    tiered = classify_tiers(ranked)
    for p in tiered:
        assert p["tier"] in ("Foundational", "Core", "Cutting Edge")


def test_classify_tiers_has_all_three():
    ranked = rank(PAPERS, interests="contrastive learning", top_n=5)
    tiered = classify_tiers(ranked)
    tiers = {p["tier"] for p in tiered}
    assert tiers == {"Foundational", "Core", "Cutting Edge"}