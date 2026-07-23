import requests
from src.config import SYNTH_MAX_TOKENS, GROQ_MODEL, GROQ_URL, GROQ_API_KEY


def synthesize(topic: str, papers: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file (free at https://console.groq.com).")
    paper_lines = []
    for i, p in enumerate(papers, 1):
        abstract_snippet = (p.get("abstract") or "")[:200]
        authors = p.get("authors", [])
        first_author = authors[0]["name"] if authors else "Unknown"
        line = (
            f"{i}. {p['title']} "
            f"({p.get('year', '?')}, {p.get('citationCount', 0)} citations, {first_author} et al.)\n"
            f"   {abstract_snippet}"
        )
        paper_lines.append(line)

    papers_block = "\n\n".join(paper_lines)

    prompt = f"""You are a research assistant helping an ML student understand a body of literature.

Topic: "{topic}"

Top papers:

{papers_block}

Write exactly 3 sentences:
1. What the main themes across these papers are.
2. What the central open question or tension in this field is.
3. Which paper to read first and why.

Be specific. Name actual papers and authors."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": SYNTH_MAX_TOKENS,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    choices = resp.json().get("choices", [])
    if not choices:
        raise ValueError(f"Groq returned no choices. Response: {resp.text[:300]}")
    return choices[0]["message"]["content"]