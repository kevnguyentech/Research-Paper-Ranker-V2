from unittest.mock import MagicMock, patch

from src.synthesize import synthesize

PAPERS = [
    {
        "title": "Test Paper",
        "abstract": "A paper about contrastive learning.",
        "year": 2023,
        "citationCount": 100,
        "authors": [{"name": "Jane Doe"}],
    }
]


@patch("src.synthesize.GROQ_API_KEY", "test-key")
@patch("src.synthesize.requests.post")
def test_synthesize_returns_string(mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "Sentence one. Sentence two. Sentence three."}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()
    result = synthesize("contrastive learning", PAPERS)
    assert isinstance(result, str)
    assert len(result) > 0