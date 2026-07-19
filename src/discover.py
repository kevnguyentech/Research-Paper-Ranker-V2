import argparse
import sys
from pathlib import Path

from src.embed import classify_tiers, rank
from src.fetch_papers import fetch_by_topic, fetch_citations, resolve_paper
from src.synthesize import synthesize
from src.config import TOP_N_DEFAULT, OUTPUT_DIR


def print_results(papers: list[dict], topic: str) -> None:
    print(f"\n{'='*60}")
    print(f"Results for: {topic}")
    print(f"{'='*60}\n")
    for i, p in enumerate(papers, 1):
        authors = p.get("authors", [])
        first_author = authors[0]["name"] if authors else "Unknown"
        print(
            f"{i:2}. [{p['tier']:12}] score={p['final_score']:.3f}\n"
            f"    {p['title']}\n"
            f"    {first_author} et al. | {p.get('year', '?')} | "
            f"{p.get('citationCount', 0)} citations | "
            f"{p.get('venue') or 'venue unknown'}\n"
        )


def export_markdown(papers: list[dict], topic: str, synthesis: str | None, path: Path) -> None:
    lines = [f"# Research: {topic}\n"]
    if synthesis:
        lines += [f"## Synthesis\n", synthesis, "\n"]
    lines.append("## Papers\n")
    for i, p in enumerate(papers, 1):
        authors = p.get("authors", [])
        first_author = authors[0]["name"] if authors else "Unknown"
        lines.append(
            f"### {i}. {p['title']}\n"
            f"**Tier:** {p['tier']} | "
            f"**Score:** {p['final_score']:.3f} | "
            f"**Year:** {p.get('year', '?')} | "
            f"**Citations:** {p.get('citationCount', 0)}\n\n"
            f"**Authors:** {first_author} et al.\n\n"
            f"{p.get('abstract', '')[:300]}...\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Research Paper Ranker v2")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--topic", type=str, help="Search by research topic")
    mode.add_argument("--seed", type=str, help="Expand from a seed paper (title or S2 ID)")

    parser.add_argument("--interests", type=str, required=True, help="Your research interests (used for ranking)")
    parser.add_argument("--direction", choices=["forward", "backward", "both"], default="both", help="Citation direction for --seed mode")
    parser.add_argument("--already-read", type=str, default="", help="Comma-separated paper titles you've already read")
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--synthesize", action="store_true", help="Generate LLM synthesis of top papers")
    parser.add_argument("--output", type=str, default="", help="Save results to a markdown file")
    parser.add_argument("--refresh", action="store_true", help="Bypass cache and re-fetch from S2")

    args = parser.parse_args()

    # 1. fetch
    if args.topic:
        print(f"Searching papers on: {args.topic}")
        papers = fetch_by_topic(args.topic, limit=100, refresh=args.refresh)
        topic_label = args.topic
    else:
        print(f"Resolving seed paper: {args.seed}")
        paper_id = resolve_paper(args.seed)
        if not paper_id:
            print(f"Could not resolve paper: {args.seed}")
            sys.exit(1)
        print(f"Fetching citations (direction={args.direction})...")
        papers = fetch_citations(paper_id, direction=args.direction, limit=100, refresh=args.refresh)
        topic_label = args.seed

    if not papers:
        print("No papers found.")
        sys.exit(1)

    print(f"Fetched {len(papers)} papers. Ranking...")

    # 2. parse already-read
    already_read = [t.strip() for t in args.already_read.split(",") if t.strip()] if args.already_read else []

    # 3. rank + tier
    ranked = rank(papers, interests=args.interests, already_read=already_read or None, top_n=args.top_n)
    tiered = classify_tiers(ranked)

    # 4. print
    print_results(tiered, topic_label)

    # 5. synthesize
    synth_text = None
    if args.synthesize:
        print("Synthesizing...\n")
        synth_text = synthesize(topic_label, tiered)
        print("=== Synthesis ===")
        print(synth_text)

    # 6. export
    if args.output:
        export_markdown(tiered, topic_label, synth_text, OUTPUT_DIR / args.output)


if __name__ == "__main__":
    main()