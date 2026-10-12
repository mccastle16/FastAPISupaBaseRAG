"""
chunk_docs.py - Step 3 of the RAG pipeline: split pages into short passages.

Reads data/processed/docs.jsonl (made by load_docs.py) and writes
data/processed/chunks.jsonl with one chunk per line:

    {"chunk_id", "doc_id", "source", "section", "title", "url",
     "headings", "text", "embed_text"}

Standard library only. Run with:  python chunk_docs.py

Why chunk at all?
  A page can be 60,000 characters. If we embed a whole page, its "meaning
  fingerprint" is an average of many topics, so it matches no question well,
  and it's too big to hand to the AI as context. Short passages about one
  thing match questions sharply and are cheap to include in a prompt.

How we chunk (per page):
  1. Parse the page into blocks: headings, paragraphs and code blocks.
     Headings inside code blocks (like "# comment" in Python) are ignored.
  2. Group blocks into sections: each heading starts a new section.
  3. Pack each section's blocks into chunks of about TARGET_CHARS, never
     cutting a paragraph or code block in half unless it's bigger than
     MAX_CHARS on its own (then we cut at line boundaries, and re-fence code).
  4. Merge tiny chunks (e.g. a heading with one sentence) into a neighbour.
  5. Give every chunk a breadcrumb ("Page title > Section > Subsection").
     It's put in front of the text for embedding, so a chunk that says
     "set it to true" still carries what "it" is.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/processed/docs.jsonl"
OUTPUT = ROOT / "data/processed/chunks.jsonl"

# Sizes are in characters. A rough rule: 1 token is about 4 characters of
# English, so 1500 chars is about 375 tokens.
TARGET_CHARS = 1500   # aim for chunks about this size
MAX_CHARS = 2500      # a chunk may grow to this to avoid cutting a block
MIN_CHARS = 300       # chunks smaller than this get merged into a neighbour

# A fence line can't contain more backticks after the fence (```x``` on one
# line is inline code), same rule as in load_docs.py.
FENCE_RE = re.compile(r"^[ \t]*(`{3,}(?=[^`\n]*$)|~{3,})", re.M)
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


# --------------------------------------------------------------------------
# 1) Parse a page into blocks
# --------------------------------------------------------------------------
# A block is a dict: {"kind": "heading" | "code" | "para", "text": str,
#                     "level": int (headings only)}

def parse_blocks(text):
    blocks, para = [], []
    lines = text.split("\n")

    def flush_para():
        if para:
            blocks.append({"kind": "para", "text": "\n".join(para).strip()})
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = FENCE_RE.match(line)
        if fence:
            # A code block runs to the matching closing fence (same character,
            # at least as long). Everything inside is kept as one block.
            flush_para()
            marker = fence[1]
            close = re.compile(rf"^[ \t]*{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            blocks.append({"kind": "code", "text": "\n".join(lines[i:j + 1])})
            i = j + 1
            continue
        heading = HEADING_RE.match(line)
        if heading:
            flush_para()
            blocks.append({"kind": "heading", "level": len(heading[1]),
                           "title": heading[2].strip(), "text": line.strip()})
        elif not line.strip():
            flush_para()          # a blank line ends a paragraph
        else:
            para.append(line)
        i += 1
    flush_para()
    return [b for b in blocks if b["text"]]


# --------------------------------------------------------------------------
# 2) Group blocks into sections (one per heading)
# --------------------------------------------------------------------------

def slugify(heading, source):
    """Approximate the #anchor each docs site generates for a heading.

    FastAPI: every run of punctuation/spaces becomes one "-"
             ("Check the `openapi.json`" -> "check-the-openapi-json").
             Matches 95% of FastAPI's real anchors.
    Supabase: GitHub style, punctuation is dropped and each space is a "-".
    """
    heading = re.sub(r"[`*_\[\]()]", "", heading).lower()
    if source == "fastapi":
        return re.sub(r"[^a-z0-9]+", "-", heading).strip("-")
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s", "-", heading).strip("-")


def split_sections(blocks, page_title, source):
    """Returns a list of sections: {"headings": [...], "anchor": str, "blocks": [...]}"""
    sections = []
    stack = []          # the current heading path, as (level, title) pairs
    current = {"headings": [], "anchor": "", "blocks": []}

    for block in blocks:
        if block["kind"] == "heading":
            if current["blocks"]:
                sections.append(current)
            # Pop headings at the same or deeper level, then push this one.
            level = block["level"]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block["title"]))
            # The page's own "# Title" heading duplicates the title field,
            # so leave it out of the breadcrumb.
            path = [t for lvl, t in stack if not (lvl == 1 and t == page_title)]
            current = {"headings": path, "anchor": slugify(block["title"], source) if level > 1 else "",
                       "blocks": [block]}
        else:
            current["blocks"].append(block)
    if current["blocks"]:
        sections.append(current)
    return sections


# --------------------------------------------------------------------------
# 3) Pack blocks into chunks
# --------------------------------------------------------------------------

def split_big_block(block):
    """Cut one oversized block into pieces <= TARGET_CHARS at line boundaries.

    Code is re-fenced so every piece is still a valid code block on its own
    (an opening ``` without a closing one would confuse the AI later).
    """
    lines = block["text"].split("\n")
    if block["kind"] == "code":
        opening, body = lines[0], lines[1:]
        closing = body.pop() if body and FENCE_RE.match(body[-1]) else "```"
    else:
        opening = closing = None
        body = lines

    pieces, piece, size = [], [], 0
    for line in body:
        if piece and size + len(line) + 1 > TARGET_CHARS:
            pieces.append(piece)
            piece, size = [], 0
        piece.append(line)
        size += len(line) + 1
    if piece:
        pieces.append(piece)

    if block["kind"] == "code":
        return ["\n".join([opening, *p, closing]) for p in pieces]
    return ["\n".join(p) for p in pieces]


def pack_section(section):
    """Greedily fill chunks with whole blocks, up to TARGET_CHARS."""
    chunks, current = [], []

    def size(parts):
        return sum(len(p) + 2 for p in parts)

    for block in section["blocks"]:
        texts = [block["text"]] if len(block["text"]) <= MAX_CHARS else split_big_block(block)
        for text in texts:
            # Start a new chunk if adding this would overshoot the target.
            # (We still allow up to MAX_CHARS rather than leave a tiny chunk.)
            if current and size(current) + len(text) > TARGET_CHARS:
                if size(current) >= MIN_CHARS or size(current) + len(text) > MAX_CHARS:
                    chunks.append(current)
                    current = []
            current.append(text)
    if current:
        chunks.append(current)

    return [{"headings": section["headings"], "anchor": section["anchor"],
             "text": "\n\n".join(parts)} for parts in chunks]


# --------------------------------------------------------------------------
# 4) Merge tiny chunks
# --------------------------------------------------------------------------

def merge_small(chunks):
    """Merge chunks under MIN_CHARS into the next chunk (or the previous one
    if it's the last), as long as the result stays under MAX_CHARS.

    The merged chunk keeps the first chunk's breadcrumb, and the second
    chunk's heading line stays in the text, so no context is lost.
    """
    merged = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        if prev and len(prev["text"]) < MIN_CHARS and len(prev["text"]) + len(chunk["text"]) <= MAX_CHARS:
            prev["text"] += "\n\n" + chunk["text"]
            STATS["merged small chunks"] += 1
        else:
            merged.append(chunk)
    # A small chunk at the very end has no next one: merge it backwards.
    if len(merged) > 1 and len(merged[-1]["text"]) < MIN_CHARS \
            and len(merged[-2]["text"]) + len(merged[-1]["text"]) <= MAX_CHARS:
        merged[-2]["text"] += "\n\n" + merged.pop()["text"]
        STATS["merged small chunks"] += 1
    return merged


# --------------------------------------------------------------------------
# 5) Build the output records
# --------------------------------------------------------------------------

STATS = Counter()


def chunk_page(doc):
    blocks = parse_blocks(doc["text"])
    sections = split_sections(blocks, doc["title"], doc["source"])
    pieces = []
    for section in sections:
        pieces.extend(pack_section(section))
    pieces = merge_small(pieces)

    records = []
    for n, piece in enumerate(pieces):
        breadcrumb = " > ".join([doc["title"], *piece["headings"]])
        records.append({
            "chunk_id": f"{doc['id']}#{n}",
            "doc_id": doc["id"],
            "source": doc["source"],
            "section": doc["section"],
            "title": doc["title"],
            # Link straight to the section, so answers can cite it precisely.
            "url": doc["url"] + (f"#{piece['anchor']}" if piece["anchor"] else ""),
            "headings": piece["headings"],
            "text": piece["text"],
            # What step 4 will embed: the breadcrumb gives the passage its
            # context (which page and section it comes from).
            "embed_text": breadcrumb + "\n\n" + piece["text"],
        })
    return records


def main():
    docs = [json.loads(line) for line in INPUT.open(encoding="utf-8")]
    chunks = []
    for doc in docs:
        chunks.extend(chunk_page(doc))

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ---------------- Report ----------------
    sizes = sorted(len(c["text"]) for c in chunks)
    print(f"Read {len(docs)} pages, saved {len(chunks)} chunks to {OUTPUT.relative_to(ROOT)}\n")
    for source in ("fastapi", "supabase"):
        n = sum(1 for c in chunks if c["source"] == source)
        print(f"  {source:9} {n:6} chunks")
    print(f"\n  chunks per page: avg {len(chunks) / len(docs):.1f}, "
          f"max {max(Counter(c['doc_id'] for c in chunks).values())}")
    print("\nChunk size in characters (about 4 chars per token):")
    print(f"  min {sizes[0]}, median {statistics.median(sizes):.0f}, "
          f"90th pct {sizes[int(len(sizes) * 0.9)]}, max {sizes[-1]}")
    buckets = Counter()
    for s in sizes:
        buckets["< 300" if s < 300 else "300-1500" if s <= 1500 else "1500-2500" if s <= 2500 else "> 2500"] += 1
    for label in ("< 300", "300-1500", "1500-2500", "> 2500"):
        print(f"  {label:10} {buckets[label]:6}")

    print("\nChecks (should be 0):")
    broken = [c["chunk_id"] for c in chunks if len(FENCE_RE.findall(c["text"])) % 2]
    print(f"  chunks with an unclosed code fence: {len(broken)}  {broken[:5]}")
    print(f"  chunks over {MAX_CHARS} chars:          {buckets['> 2500']}")
    if STATS:
        print(f"\n{dict(STATS)}")


if __name__ == "__main__":
    main()
