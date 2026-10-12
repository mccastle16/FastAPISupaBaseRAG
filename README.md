# FastAPISupaBaseRAG

A RAG assistant that answers developer questions about FastAPI and Supabase with cited links to the source docs.

```powershell
python ask.py "How do I add GitHub login to my Supabase app?"
```

The answer streams in with `[1]`, `[2]`... citations, followed by the list of
source passages with links to the exact doc sections.

RAG (retrieval-augmented generation) means: find the doc passages that answer
the question, give them to Claude, and have Claude answer *only* from them,
citing each one. Answers stay current with the docs, can be checked against
the source, and the assistant can say "the docs don't cover this" instead of
guessing.

## How it works

The pipeline is seven steps, one script each. Every script reads the previous
step's output from `data/processed/`.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1-2. Load and clean | `load_docs.py` | Reads the raw FastAPI `.md` and Supabase `.mdx` docs, fills in code examples and shared snippets, strips markup outside code, makes links absolute | `docs.jsonl` (932 pages) |
| 3. Chunk | `chunk_docs.py` | Splits pages at their headings into ~1,500-character passages, never cutting a code block, and adds a "Page > Section" breadcrumb to each | `chunks.jsonl` (7,503 chunks) |
| 4. Embed | `embed_chunks.py` | Turns each passage into a 384-number "meaning fingerprint" with a local model (`BAAI/bge-small-en-v1.5`) | `embeddings.npy` |
| 5. Search | `search.py` | Embeds the question the same way and returns the closest passages | (in memory) |
| 6. Generate | `ask.py` | Gives the best 6 passages to Claude (Opus 5.5), which answers with `[n]` citations | answer + sources |
| 7. Evaluate | `evaluate.py` | Runs 48 test questions through search and scores whether the right pages come back | `eval/results/` |

## Setup

Requires Python 3.11 or newer. These commands are for Windows PowerShell.

**1. Create a virtual environment and install the packages**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`load_docs.py` and `chunk_docs.py` use only the standard library. The rest
need `sentence-transformers` (includes PyTorch, a few hundred MB) and
`anthropic`.

**2. Download the raw docs**

The docs come straight from the official GitHub repos. These are shallow,
sparse clones, so only the docs folders are downloaded, not the full repos:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git data/fastapi
git -C data/fastapi sparse-checkout set docs/en/docs docs_src

git clone --depth 1 --filter=blob:none --sparse https://github.com/supabase/supabase.git data/supabase
git -C data/supabase sparse-checkout set apps/docs/content/guides apps/docs/content/troubleshooting apps/docs/content/_partials
```

Everything under `data/` is ignored by git: the raw docs and the generated
files can always be rebuilt.

**3. Add your Anthropic API key** (only needed for `ask.py`)

Copy `.env.example` to `.env` and put your key in it. Create a key at
<https://console.anthropic.com/settings/keys>. `.env` is ignored by git, so the
key is never committed.

## Build the index

Run once, and again whenever you update the docs or change how they're
processed:

```powershell
python load_docs.py      # about 6 seconds
python chunk_docs.py     # a few seconds
python embed_chunks.py   # 5-20 minutes on a CPU; downloads the model (~130 MB) the first time
```

Each script prints a report: counts, sizes, and checks that should be 0.
`embed_chunks.py` skips the work if the chunks haven't changed since the last
run (add `--force` to redo it).

## Ask questions

```powershell
python ask.py "How do I add GitHub login to my Supabase app?"
python ask.py "How do I verify a Supabase JWT in a FastAPI endpoint?"
python ask.py              # interactive: ask several questions, empty line to quit
```

Each answer streams in as it's written, followed by the sources (cited ones
marked `*`) and what the question cost: usually 2-4 US cents.

Options:

- `--source fastapi` or `--source supabase`: use only one set of docs.
- `--balance`: always take passages from both sources. By default this
  happens only when the question names both FastAPI and Supabase (the setting
  that scored best in evaluation).
- `-k 8`: give Claude more passages (default 6).

To see what search finds without calling Claude (free):

```powershell
python search.py "How do I add GitHub login?"
python search.py           # interactive
```

The first search in a session takes up to a minute while the embedding model
loads; after that each search takes well under a second.

## Evaluate

```powershell
python evaluate.py
```

Runs every question in `eval/questions.jsonl` through search and checks
whether the page that answers it comes back. It makes no API calls, so it's
free to run after every change. Each run is saved to `eval/results/`, and the
report shows how much each score went up or down since the previous run.

Current results (k = 6, `auto` balancing):

| Questions | hit@1 | hit@3 | hit@6 |
|---|---|---|---|
| All 45 answerable | 62% | 78% | **93%** |
| FastAPI (19) | 58% | 79% | 89% |
| Supabase (22) | 68% | 77% | 95% |
| Need both sources (4) | 50% | 75% | 100% |

hit@6 is the share of questions where the right page is among the 6 passages
Claude sees. If it's missing, Claude can't answer from the docs.

To add a test question, add a line to `eval/questions.jsonl` with the page id(s)
that answer it (ids are the `doc_id` values in `data/processed/chunks.jsonl`).
The script checks that every id exists.

## Project layout

```
load_docs.py        step 1-2: load and clean the raw docs
chunk_docs.py       step 3: split pages into passages
embed_chunks.py     step 4: embed the passages
search.py           step 5: find the passages for a question
ask.py              step 6: answer with Claude
evaluate.py         step 7: score retrieval
eval/
  questions.jsonl   test questions and the pages that answer them
  results/          one file per evaluation run
data/               (not in git)
  fastapi/          raw FastAPI docs
  supabase/         raw Supabase docs
  processed/        docs.jsonl, chunks.jsonl, embeddings.npy, embeddings_meta.json
```

## Known limitations

- **Ambiguous questions pick the wrong product.** "How do I return a 404
  error?" finds Supabase's Edge Function 404 page instead of FastAPI's error
  handling page. Naming the product ("...in FastAPI") fixes it.
- **Long chunks are partly unseen by search.** 284 chunks (3.8%), mostly
  code-heavy Supabase pages, are longer than the embedding model's 512-token
  limit. Search only matches on their first 512 tokens; Claude still gets the
  full text.
- **Similarity scores can't detect unanswerable questions.** Questions the docs
  don't cover score only slightly lower than real ones, so the system prompt's
  rule to say "the docs don't cover this" is what handles them.

## Ideas for next steps

- **Answer evaluation:** run `ask()` on the test questions and check that
  Claude cites the right pages and admits when the docs don't cover something
  (about $1.50 per full run).
- **Hybrid search:** combine meaning-based search with keyword search so exact
  terms like "404" or error codes count for more.
- **A FastAPI web API** around `ask()`, so a website or app can use it.
- **Supabase pgvector:** store the vectors in Supabase instead of in memory.
