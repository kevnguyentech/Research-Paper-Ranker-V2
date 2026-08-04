import hashlib

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
        vecs = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.random(8).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(vecs)

    with patch("src.embed.encode", side_effect=_fake):
        yield