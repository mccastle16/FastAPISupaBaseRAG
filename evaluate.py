"""
evaluate.py - Step 7 of the RAG pipeline: grade whether search finds the right pages.

Runs every question in eval/questions.jsonl through search() and checks
whether the pages that answer it came back, and how high they ranked.
No API calls, so it's free: run it after every change and compare scores.

Run:  .venv\\Scripts\\python.exe evaluate.py

Each run is saved to eval/results/, and the report shows the change since the
previous run, so you can tell whether a change (chunk size, embedding model,
search settings) made retrieval better or worse.

The metrics (k = 6, the number of passages ask.py gives Claude):
  hit@1   the right page was the top result
  hit@3   the right page was in the top 3
  hit@6   the right page was somewhere in what Claude sees. This matters most:
          if it's not here, Claude can't answer from the docs.
  MRR     "mean reciprocal rank": 1 for rank 1, 1/2 for rank 2, 1/3 for rank 3,
          ... 0 if missing, averaged. One number that rewards ranking high.
  both    (cross questions only) at least one right page from EACH source
          was found. A FastAPI + Supabase question needs both halves.
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "eval/questions.jsonl"
RESULTS_DIR = ROOT / "eval/results"
CHUNKS = ROOT / "data/processed/chunks.jsonl"

K = 6   # same as ask.py's DEFAULT_K

# The search settings to compare. Add your own here to test an idea.
CONFIGS = {
    "default": {},
    "balance": {"balance": True},
    "auto": {"balance": "auto"},   # balance only when both names are in the question
}


def load_questions():
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8") if line.strip()]
    # Catch typos in expected page ids up front: a misspelled id would
    # silently count as a miss forever.
    known = {json.loads(line)["doc_id"] for line in CHUNKS.open(encoding="utf-8")}
    bad = [(q["id"], d) for q in questions for d in q["expected"] if d not in known]
    if bad:
        raise SystemExit("Unknown page ids in eval/questions.jsonl:\n"
                         + "\n".join(f"  {qid}: {d}" for qid, d in bad))
    return questions


def grade(question, results):
    """Score one question's results."""
    expected = set(question["expected"])
    returned = [r["doc_id"] for r in results]
    rank = next((i + 1 for i, d in enumerate(returned) if d in expected), None)
    found_sources = {d.split(":")[0] for d in returned if d in expected}
    return {
        "id": question["id"],
        "category": question["category"],
        "rank": rank,                                   # None = not found in top K
        "both_sources": found_sources == {"fastapi", "supabase"},
        "top_score": results[0]["score"] if results else None,
        "returned": [(r["doc_id"], round(r["score"], 3)) for r in results],
    }


def summarize(graded):
    """Average the metrics over a list of graded questions."""
    n = len(graded)
    if not n:
        return {}
    ranks = [g["rank"] for g in graded]
    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        f"hit@{K}": sum(1 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def previous_run():
    runs = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


def fmt_delta(now, before):
    if before is None:
        return ""
    d = now - before
    return "" if abs(d) < 0.005 else f" ({'+' if d > 0 else ''}{d * 100:.0f})"


def main():
    questions = load_questions()
    answerable = [q for q in questions if q["category"] != "unanswerable"]
    unanswerable = [q for q in questions if q["category"] == "unanswerable"]

    from search import load_index, search   # slow import (loads the model)
    print("Loading search index and model...")
    load_index()

    run = {"time": datetime.now().isoformat(timespec="seconds"), "k": K, "configs": {}}
    start = time.time()
    for name, settings in CONFIGS.items():
        graded = [grade(q, search(q["question"], k=K, **settings)) for q in questions]
        by_cat = {}
        for cat in ("fastapi", "supabase", "cross"):
            by_cat[cat] = summarize([g for g in graded if g["category"] == cat])
        cross = [g for g in graded if g["category"] == "cross"]
        run["configs"][name] = {
            "overall": summarize([g for g in graded if g["category"] != "unanswerable"]),
            "by_category": by_cat,
            "cross_both_sources": sum(g["both_sources"] for g in cross) / len(cross) if cross else None,
            "questions": graded,
        }
    seconds = time.time() - start

    # ---------------- Report ----------------
    before = previous_run()
    print(f"\n{len(answerable)} answerable + {len(unanswerable)} unanswerable questions, "
          f"k={K}, {seconds:.0f}s")
    if before:
        print(f"(numbers in brackets: change in points since the run at {before['time']})")

    metrics = ["hit@1", "hit@3", f"hit@{K}", "mrr"]
    for name, result in run["configs"].items():
        old = (before or {}).get("configs", {}).get(name)
        print(f"\n=== {name} {CONFIGS[name] or ''}")
        print(f"  {'':14}" + "".join(f"{m:>7}{'':5}" for m in metrics))
        rows = [("overall", result["overall"], (old or {}).get("overall"))]
        rows += [(cat, s, ((old or {}).get("by_category") or {}).get(cat))
                 for cat, s in result["by_category"].items()]
        for label, s, o in rows:
            cells = "".join(f"{s[m]:>7.0%}{fmt_delta(s[m], (o or {}).get(m)):<5}" if m != "mrr"
                            else f"{s[m]:>7.2f}{fmt_delta(s[m], (o or {}).get(m)):<5}"
                            for m in metrics)
            print(f"  {label + ' (' + str(s['n']) + ')':14}{cells}")
        if result["cross_both_sources"] is not None:
            print(f"  cross questions with pages from both sources: {result['cross_both_sources']:.0%}"
                  f"{fmt_delta(result['cross_both_sources'], (old or {}).get('cross_both_sources'))}")

    # Misses for the default config: these are the ones to look at.
    default = run["configs"]["default"]["questions"]
    q_by_id = {q["id"]: q for q in questions}
    misses = [g for g in default if g["category"] != "unanswerable" and not g["rank"]]
    print(f"\nMisses with default settings ({len(misses)}): the right page wasn't in the top {K}")
    for g in misses:
        q = q_by_id[g["id"]]
        print(f"\n  {g['id']}: {q['question']}")
        print(f"    expected: {', '.join(q['expected'])}")
        for doc_id, score in g["returned"][:3]:
            print(f"    got:      {score:.3f}  {doc_id}")

    # Could a score threshold tell answerable from unanswerable questions?
    ans_scores = [g["top_score"] for g in default if g["category"] != "unanswerable"]
    na = [g for g in default if g["category"] == "unanswerable"]
    print("\nTop result score (could a cut-off detect questions the docs don't cover?)")
    print(f"  answerable:   min {min(ans_scores):.3f}, median {statistics.median(ans_scores):.3f}, "
          f"max {max(ans_scores):.3f}")
    for g in na:
        print(f"  unanswerable: {g['top_score']:.3f}  {q_by_id[g['id']]['question']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(run, indent=1), encoding="utf-8", newline="\n")
    print(f"\nSaved this run to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
