import json
from unittest.mock import MagicMock, patch

from src.fetch_papers import fetch_by_topic, fetch_citations, resolve_paper

MOCK_PAPER = {
    "paperId": "abc123",
    "title": "Test Paper",
    "abstract": "A test abstract about machine learning.",
    "year": 2023,
    "venue": "NeurIPS",
    "citationCount": 100,
    "authors": [{"name": "Jane Doe"}],
}


def _mock_get(return_data: dict):
    mock = MagicMock()
    mock.return_value.json.return_value = return_data
    mock.return_value.raise_for_status = MagicMock()
    mock.return_value.status_code = 200
    return mock


@patch("src.fetch_papers.requests.get")
def test_fetch_by_topic_returns_papers(mock_get):
    mock_get.side_effect = _mock_get({"data": [MOCK_PAPER]})
    papers = fetch_by_topic("machine learning", limit=1, refresh=True)
    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper"


@patch("src.fetch_papers.requests.get")
def test_fetch_by_topic_filters_no_abstract(mock_get):
    no_abstract = {**MOCK_PAPER, "abstract": None}
    mock_get.side_effect = _mock_get({"data": [no_abstract, MOCK_PAPER]})
    papers = fetch_by_topic("machine learning", limit=2, refresh=True)
    assert len(papers) == 1


@patch("src.fetch_papers.requests.get")
def test_resolve_paper_returns_id(mock_get):
    mock_get.side_effect = _mock_get({"data": [MOCK_PAPER]})
    pid = resolve_paper("Test Paper", )
    assert pid == "abc123"


def test_resolve_paper_passthrough_hex_id():
    hex_id = "a" * 40
    result = resolve_paper(hex_id)
    assert result == hex_id


@patch("src.fetch_papers.requests.get")
def test_fetch_citations_forward(mock_get):
    mock_get.side_effect = _mock_get({"data": [{"citingPaper": MOCK_PAPER}]})
    papers = fetch_citations("abc123", direction="forward", refresh=True)
    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper"


@patch("src.fetch_papers.requests.get")
def test_fetch_citations_backward(mock_get):
    mock_get.side_effect = _mock_get({"data": [{"citedPaper": MOCK_PAPER}]})
    papers = fetch_citations("abc123", direction="backward", refresh=True)
    assert len(papers) == 1