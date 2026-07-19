import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR   = Path(__file__).parent.parent
CACHE_DIR  = ROOT_DIR / "data" / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Semantic Scholar
BASE_URL       = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS   = "paperId,title,abstract,year,venue,citationCount,authors"
CACHE_TTL_DAYS = 7
REQUEST_DELAY  = 1.1   # seconds; drops to 0.15 if S2_API_KEY is set

# SBERT
SBERT_MODEL = "all-MiniLM-L6-v2"

# Ranking
NOVELTY_LAMBDA  = 0.3
TOP_N_DEFAULT   = 10

# Anthropic
SYNTH_MODEL      = "claude-sonnet-4-6"
SYNTH_MAX_TOKENS = 300

# API keys
S2_API_KEY       = os.getenv("S2_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")