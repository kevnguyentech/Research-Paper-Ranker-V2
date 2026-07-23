# Research Paper Ranker v2

A CLI tool that helps you find and understand relevant research papers. Give it a topic or a seed paper, and it returns a ranked reading list with tier labels and an optional synthesis of what the field is actually about.

Built on top of the Semantic Scholar API and SBERT embeddings.

## What it does

- Search papers by topic or research question
- Expand from a seed paper using its citation graph (forward and backward)
- Rank results by relevance to your interests
- Penalize papers too similar to ones you have already read (novelty scoring)
- Classify papers into Foundational, Core, and Cutting Edge tiers
- Generate a 3-sentence synthesis of the top results using an LLM

## How it is different from v1

Version 1 was author-centric. You needed to know who you were looking for before you started. Version 2 is topic-first. You describe what you want to learn and the tool finds the papers.

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/kevnguyentech/Research-Paper-Ranker-V2
cd Research-Paper-Ranker-V2
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Mac/Linux
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```env
S2_API_KEY=your_semantic_scholar_key
GROQ_API_KEY=your_groq_key
```

Both are free:
- Semantic Scholar API key: https://www.semanticscholar.org/product/api
- Groq API key: https://console.groq.com

The S2 key raises your rate limit from 1 request/second to 10. Without it the tool works but runs slower.

## Usage

**Topic mode** -- search by research question:

```bash
python -m src.discover \
  --topic "contrastive learning" \
  --interests "self-supervised visual representation"
```

**Seed mode** -- expand from a paper you already know:

```bash
python -m src.discover \
  --seed "Attention is all you need" \
  --interests "transformers NLP" \
  --direction backward
```

**With all options:**

```bash
python -m src.discover \
  --topic "meta-learning few-shot" \
  --interests "fast adaptation low data regimes" \
  --already-read "paper title one, paper title two" \
  --top-n 10 \
  --synthesize \
  --output results.md
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | | Search by topic. Required if not using `--seed` |
| `--seed` | | Seed paper title or S2 paper ID |
| `--interests` | | Your research interests, used for ranking. Required |
| `--direction` | `both` | Citation direction for seed mode: `forward`, `backward`, or `both` |
| `--already-read` | | Comma-separated titles of papers you have read. Lowers their score |
| `--top-n` | `10` | Number of results to show |
| `--synthesize` | off | Generate a 3-sentence LLM synthesis of the top results |
| `--output` | | Save results as a markdown file inside `outputs/` |
| `--refresh` | off | Bypass cache and re-fetch from Semantic Scholar |

## How ranking works

Each paper gets a relevance score based on cosine similarity between its embedding (title + abstract) and your stated interests, using the `all-MiniLM-L6-v2` SBERT model.

If you pass `--already-read`, the score is adjusted:        

final score = relevance - 0.3 * max similarity to already-read papers

This pushes down papers that cover ground you have already seen and surfaces ones that are genuinely new to you.

## How tiers work

Papers are bucketed by citation count percentile within the result set, not against fixed global thresholds. This makes it self-normalizing across fields.

- Top 25% by citations: **Foundational**
- Middle 50%: **Core**
- Bottom 25%: **Cutting Edge**

A paper with 50 citations can still be Foundational if everything else in the set has fewer.

## Running tests

```bash
python -m pytest tests/ -v
```

## Project structure

```bash
Research-Paper-Ranker-V2/
├── src/
│   ├── __init__.py
│   ├── config.py          # constants, env vars, path setup
│   ├── fetch_papers.py    # Semantic Scholar API client with disk cache
│   ├── embed.py           # SBERT encoding, novelty scoring, tier classification
│   ├── synthesize.py      # LLM synthesis via Groq
│   └── discover.py        # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_fetch.py
│   ├── test_embed.py
│   └── test_synthesize.py
├── data/
│   └── cache/             # JSON cache files, auto-created, TTL 7 days
├── outputs/               # markdown exports
├── conftest.py            # makes src importable during pytest
├── .env                   # API keys, gitignored
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
## Cache

Every API response is cached to `data/cache/` for 7 days. Repeat runs on the same topic are instant. Pass `--refresh` to force a fresh fetch.