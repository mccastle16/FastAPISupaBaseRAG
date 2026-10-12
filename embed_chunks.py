"""
embed_chunks.py - Step 4 of the RAG pipeline: give each chunk a "meaning fingerprint".

Reads data/processed/chunks.jsonl (made by chunk_docs.py) and turns each
chunk's embed_text into a vector (a list of 384 numbers) with a local model.
Writes:
    data/processed/embeddings.npy        one row per chunk, same order as chunks.jsonl
    data/processed/embeddings_meta.json  which model made them, and how to embed questions

Needs:  pip install sentence-transformers   (into .venv)
Run:    .venv\\Scripts\\python.exe embed_chunks.py          (add --force to redo)

What is an embedding?
  The model reads a passage and outputs a point in a 384-dimensional space,
  placed so that passages with similar *meaning* land close together, even
  if they use different words ("delete a row" ~ "remove a record").
  At question time we embed the question the same way and look for the
  nearest chunks. That's step 5.

Why normalize?
  We scale every vector to length 1. Then "how close are two vectors"
  (cosine similarity) is just a dot product, which is fast and simple.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data/processed/chunks.jsonl"
VECTORS = ROOT / "data/processed/embeddings.npy"
META = ROOT / "data/processed/embeddings_meta.json"

# A small, fast English retrieval model (384 dimensions, ~130 MB download,
# cached by Hugging Face after the first run).
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models are trained so that *questions* start with this instruction and
# *passages* don't. Getting this wrong quietly lowers retrieval quality, so we
# save it in the meta file for the search step to reuse.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BATCH_SIZE = 32   # chunks embedded at once; lower it if you run out of memory

# A few questions to sanity-check the vectors at the end.
TEST_QUESTIONS = [
    "How do I run a FastAPI app in development mode?",
    "How do I add GitHub login to my Supabase project?",
    "Should I use HNSW or IVFFlat indexes for pgvector?",
]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    force = "--force" in sys.argv
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    chunks_hash = file_sha256(CHUNKS)

    # Skip the work if nothing changed since the last run. Embedding takes
    # minutes; checking a hash takes milliseconds.
    if not force and VECTORS.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta.get("chunks_sha256") == chunks_hash and meta.get("model") == MODEL_NAME:
            print("Embeddings are already up to date with chunks.jsonl (use --force to redo).")
            return

    print(f"Loading model {MODEL_NAME} (downloads once, then cached)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    max_tokens = model.max_seq_length

    # The model only reads the first `max_tokens` tokens of each text and
    # silently ignores the rest. Count how often that happens, so we know
    # whether our chunk size fits the model.
    texts = [c["embed_text"] for c in chunks]
    lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]]
    too_long = [(n, c["chunk_id"]) for n, c in zip(lengths, chunks) if n > max_tokens]

    print(f"Embedding {len(texts)} chunks on CPU...")
    start = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,      # length 1, so cosine similarity = dot product
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    seconds = time.time() - start

    np.save(VECTORS, vectors)
    META.write_text(json.dumps({
        "model": MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "normalized": True,
        "query_prefix": QUERY_PREFIX,
        "max_tokens": max_tokens,
        "chunks_file": CHUNKS.name,
        "chunks_sha256": chunks_hash,
        # Row i of embeddings.npy belongs to line i of chunks.jsonl. These
        # ids let later steps double-check that the two files still match.
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }, indent=1), encoding="utf-8", newline="\n")

    # ---------------- Report ----------------
    lengths.sort()
    print(f"\nSaved {vectors.shape[0]} vectors x {vectors.shape[1]} dimensions "
          f"to {VECTORS.relative_to(ROOT)} ({VECTORS.stat().st_size / 1e6:.1f} MB)")
    print(f"Took {seconds / 60:.1f} min ({len(texts) / seconds:.0f} chunks/sec)")
    print(f"\nTokens per chunk: median {lengths[len(lengths) // 2]}, "
          f"90th pct {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    print(f"Chunks longer than the model's {max_tokens}-token limit (end gets cut off): "
          f"{len(too_long)} ({len(too_long) / len(texts):.1%})")
    for n, chunk_id in sorted(too_long, reverse=True)[:5]:
        print(f"  {n:5} tokens  {chunk_id}")

    # Sanity check: embed a few questions and show the nearest chunks.
    # This is a tiny preview of step 5 (search).
    print("\nSanity check, the 3 nearest chunks for a few questions:")
    questions = model.encode([QUERY_PREFIX + q for q in TEST_QUESTIONS],
                             normalize_embeddings=True, convert_to_numpy=True)
    scores = questions @ vectors.T          # one row of similarity scores per question
    for question, row in zip(TEST_QUESTIONS, scores):
        print(f"\n  Q: {question}")
        for i in np.argsort(-row)[:3]:
            print(f"     {row[i]:.3f}  {chunks[i]['chunk_id']}")


if __name__ == "__main__":
    main()
