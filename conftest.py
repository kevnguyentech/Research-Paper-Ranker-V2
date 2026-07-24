import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def no_fetch_sleep():
    """Patch time.sleep in fetch_papers so tests don't wait on rate-limit delays."""
    with patch("src.fetch_papers.time.sleep"):
        yield


@pytest.fixture(autouse=True)
def mock_sbert_encode():
    """Return deterministic fake embeddings -- avoids loading/downloading the SBERT model."""
    def _fake(texts):
        rng = np.random.default_rng(0)
        arr = rng.random((len(texts), 8)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-9)

    with patch("src.embed.encode", side_effect=_fake):
        yield