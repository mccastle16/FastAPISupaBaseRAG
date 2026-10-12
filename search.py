"""
search.py - Step 5 of the RAG pipeline: find the chunks that best answer a question.

Loads the vectors made by embed_chunks.py and searches them in memory.

Use it from the command line:
    .venv\\Scripts\\python.exe search.py "How do I add GitHub login?"
    .venv\\Scripts\\python.exe search.py "path parameters" -k 3 --source fastapi
    .venv\\Scripts\\python.exe search.py "FastAPI endpoint with Supabase JWT" --balance
    .venv\\Scripts\\python.exe search.py            (interactive: ask several questions)

Or from Python (step 6 will do this):
    from search import search
    results = search("How do I add GitHub login?", k=5)

How it works:
  Every chunk is a point in 384-dimensional space (embeddings.npy). We embed
  the question into the same space and score every chunk by how close it is.
  Because all vectors have length 1, closeness (cosine similarity) is just a
  dot product, and one matrix multiplication scores all 7,500 chunks at once
  in about a millisecond. At this size we don't need a special index; a
  database like pgvector becomes worth it at hundreds of thousands of chunks.
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data/processed/chunks.jsonl"
VECTORS = ROOT / "data/processed/embeddings.npy"
META = ROOT / "data/processed/embeddings_meta.json"

DEFAULT_K = 5
# At most this many chunks from the same page. Without a cap, one long page
# that mentions the topic everywhere can fill every slot and hide other pages
# that answer a different part of the question.
MAX_PER_PAGE = 2
# Skip a result whose vector is this similar to one we already picked.
# Supabase reuses the same snippet on many pages (e.g. every social login
# provider), and handing the AI four copies of one paragraph wastes space.
NEAR_DUPLICATE = 0.95


@lru_cache(maxsize=1)
def load_index():
    """Load chunks, vectors and the model once, then reuse them (lru_cache)."""
    if not VECTORS.exists() or not META.exists():
        raise SystemExit("No embeddings found. Run embed_chunks.py first.")

    meta = json.loads(META.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    vectors = np.load(VECTORS)

    # Safety check: row i of the vectors must belong to line i of the chunks.
    # If you re-run chunk_docs.py without re-running embed_chunks.py, they
    # no longer line up, and search would quietly return the wrong text.
    chunks_hash = hashlib.sha256(CHUNKS.read_bytes()).hexdigest()
    if (len(chunks) != len(vectors)
            or meta.get("chunks_sha256") != chunks_hash
            or meta.get("chunk_ids") != [c["chunk_id"] for c in chunks]):
        raise SystemExit("chunks.jsonl changed since the embeddings were made. "
                         "Re-run embed_chunks.py.")

    # Imported here so `import search` stays fast until a search is needed.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(meta["model"], device="cpu")

    sources = np.array([c["source"] for c in chunks])
    return chunks, vectors, sources, model, meta


def search(question, k=DEFAULT_K, source=None, balance=False, max_per_page=MAX_PER_PAGE):
    """Return the k best chunks for `question`, best first.

    Each result is the chunk's dict plus a "score" (cosine similarity, higher
    is closer).
      source:  "fastapi" or "supabase" to search only one of them.
      balance: take the best half from each source, so a question that spans
               both ("protect a FastAPI endpoint with a Supabase JWT") isn't
               answered by only one side.
    """
    chunks, vectors, sources, model, meta = load_index()

    # Embed the question exactly the way the model expects: with the query
    # prefix (passages were embedded without it) and normalized to length 1.
    query = model.encode(meta["query_prefix"] + question,
                         normalize_embeddings=True, convert_to_numpy=True)
    scores = vectors @ query                  # one score per chunk

    if balance and not source:
        # Run the same ranking once per source, then merge by score.
        half = (k + 1) // 2
        picked = (_top(scores, sources == "fastapi", half, vectors, chunks, max_per_page)
                  + _top(scores, sources == "supabase", k - half, vectors, chunks, max_per_page))
        picked.sort(key=lambda i: -scores[i])
    else:
        allowed = sources == source if source else np.ones(len(chunks), dtype=bool)
        picked = _top(scores, allowed, k, vectors, chunks, max_per_page)

    return [{**chunks[i], "score": float(scores[i])} for i in picked]


def _top(scores, allowed, k, vectors, chunks, max_per_page):
    """Indexes of the k best allowed chunks, skipping over-represented pages
    and near-duplicates of chunks already picked."""
    picked, per_page = [], {}
    for i in np.argsort(-scores):             # best score first
        if len(picked) == k:
            break
        if not allowed[i]:
            continue
        doc_id = chunks[i]["doc_id"]
        if per_page.get(doc_id, 0) >= max_per_page:
            continue
        # Same text on another page? Its vector will be almost identical.
        if picked and float(np.max(vectors[picked] @ vectors[i])) > NEAR_DUPLICATE:
            continue
        per_page[doc_id] = per_page.get(doc_id, 0) + 1
        picked.append(int(i))
    return picked


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def print_results(question, results):
    print(f"\nQ: {question}\n")
    for rank, r in enumerate(results, 1):
        breadcrumb = " > ".join([r["title"], *r["headings"]])
        snippet = " ".join(r["text"].split())[:220]
        print(f"{rank}. [{r['score']:.3f}] ({r['source']}) {breadcrumb}")
        print(f"   {r['url']}")
        print(f"   {snippet}...\n")


def main():
    parser = argparse.ArgumentParser(description="Search the FastAPI + Supabase docs.")
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="number of results")
    parser.add_argument("--source", choices=["fastapi", "supabase"], help="search one source only")
    parser.add_argument("--balance", action="store_true", help="take results from both sources")
    args = parser.parse_args()

    print("Loading index and model...")
    load_index()

    if args.question:
        print_results(args.question, search(args.question, args.k, args.source, args.balance))
        return

    print("Ask a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print_results(question, search(question, args.k, args.source, args.balance))


if __name__ == "__main__":
    main()
