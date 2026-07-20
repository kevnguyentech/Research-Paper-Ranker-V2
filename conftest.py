import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def no_fetch_sleep():
    """Patch time.sleep in fetch_papers so tests don't wait on rate-limit delays."""
    with patch("src.fetch_papers.time.sleep"):
        yield