"""
ask.py - Step 6 of the RAG pipeline: answer a question with Claude, using the docs.

    retrieve (search.py)  ->  build a prompt with the passages  ->  Claude answers

Use it from the command line:
    .venv\\Scripts\\python.exe ask.py "How do I add GitHub login to my Supabase app?"
    .venv\\Scripts\\python.exe ask.py "How do I verify a Supabase JWT in FastAPI?"
    .venv\\Scripts\\python.exe ask.py            (interactive: ask several questions)

Or from Python:
    from ask import ask
    result = ask("How do I add GitHub login?")
    print(result["answer"])

Needs an Anthropic API key in a .env file next to this script:
    ANTHROPIC_API_KEY=sk-ant-...
(.env is in .gitignore, so the key never gets committed.)

Why RAG instead of just asking Claude?
  Claude already knows a lot about FastAPI and Supabase, but that knowledge
  has a cut-off date and can be wrong on details. By pasting the relevant,
  current doc passages into the prompt and telling Claude to answer only from
  them, answers are up to date, checkable (every claim cites a passage), and
  "I don't know" is possible when the docs don't cover something.
"""

import argparse
import os
import re
import sys

import anthropic
from dotenv import load_dotenv

from search import load_index, search

MODEL = "claude-opus-5-5"
# Effort controls how much Claude thinks before answering. "medium" is plenty
# for doc Q&A; "high" helps on hard multi-step questions but costs more.
EFFORT = "medium"
DEFAULT_K = 6               # passages handed to Claude per question
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}   # Claude Opus 5.5, US$ per 1M tokens

# The system prompt sets the rules. The key ones for RAG: answer only from
# the excerpts, cite them, and admit when they don't contain the answer.
SYSTEM_PROMPT = """\
You are a documentation assistant for FastAPI and Supabase. You answer \
developers' questions using the documentation excerpts provided with each \
question.

How to answer:
- Base your answer only on the excerpts. They come from the current official \
docs, which may differ from what you remember, so prefer them when they disagree.
- Cite the excerpts you use with their number in square brackets, like [2], \
right after the sentence that relies on them. Cite only excerpts you used.
- If the excerpts don't contain the answer, say so plainly and briefly say \
what they do cover. Don't fill the gap from memory or guess. If they answer \
part of the question, answer that part and say which part is missing.
- Include code when it helps, adapted from the excerpts. Keep the answer \
focused: lead with the direct answer, then the details someone needs to act \
on it."""


def build_prompt(question, passages):
    """Put the passages first and the question last.

    Each passage is wrapped in tags with a number, a breadcrumb title and its
    URL, so Claude can tell where one ends and the next begins, and can cite
    them as [n]. Long reference material before the question tends to give
    better answers than the other way round.
    """
    parts = ["<documents>"]
    for n, p in enumerate(passages, 1):
        title = " > ".join([p["title"], *p["headings"]])
        parts.append(
            f'<document index="{n}">\n'
            f"<source>{p['source']}: {title}</source>\n"
            f"<url>{p['url']}</url>\n"
            f"<content>\n{p['text']}\n</content>\n"
            f"</document>"
        )
    parts.append("</documents>")
    parts.append(f"<question>\n{question}\n</question>")
    return "\n\n".join(parts)


def get_client():
    load_dotenv()   # reads ANTHROPIC_API_KEY from .env into the environment
    # The SDK only notices a missing key when the first request fails, which
    # would be after the slow search-model load. Check up front instead.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit(
            "No Anthropic API key found. Create a file named .env in the project "
            "folder containing:\n    ANTHROPIC_API_KEY=sk-ant-...\n"
            "You can create a key at https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic()


def ask(question, k=DEFAULT_K, source=None, balance="auto", stream_to=None, client=None):
    """Retrieve passages, ask Claude, and return the answer with its sources.

    If `stream_to` is a file (e.g. sys.stdout), the answer is printed there
    as it's written. Returns a dict with answer, passages, cited (the passage
    numbers the answer cites), usage and cost.

    balance="auto" takes passages from both sources only when the question
    names both FastAPI and Supabase (the best setting in evaluate.py).
    """
    client = client or get_client()
    passages = search(question, k=k, source=source, balance=balance)

    # Stream the response: the answer appears as it's generated instead of
    # after a long wait, and long requests can't hit an HTTP timeout.
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
        output_config={"effort": EFFORT},
        # If a safety classifier wrongly declines a harmless question, retry
        # it on Anthropic's recommended fallback model instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            if stream_to:
                print(text, end="", file=stream_to, flush=True)
        message = stream.get_final_message()

    # Claude thinks before answering (thinking blocks, hidden by default);
    # the answer is the text blocks.
    answer = "".join(b.text for b in message.content if b.type == "text")
    if message.stop_reason == "refusal":
        answer += "\n\n[Claude declined to answer this question.]"
    elif message.stop_reason == "max_tokens":
        answer += "\n\n[Answer cut off: hit the max_tokens limit.]"

    usage = message.usage
    cost = (usage.input_tokens * PRICE_PER_MTOK["input"]
            + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(passages)})
    return {
        "question": question,
        "answer": answer,
        "passages": passages,
        "cited": cited,
        "model": message.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost,
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def print_answer(question, args, client):
    print(f"\nQ: {question}\n")
    try:
        result = ask(question, k=args.k, source=args.source, balance=args.balance,
                     stream_to=sys.stdout, client=client)
    except anthropic.AuthenticationError:
        raise SystemExit("\nThe API key in .env was rejected. Check that it's correct and active.")
    except anthropic.RateLimitError:
        print("\n[Rate limited. Wait a moment and try again.]")
        return
    except anthropic.APIStatusError as e:
        print(f"\n[API error {e.status_code}: {e.message}]")
        return
    except anthropic.APIConnectionError:
        print("\n[Couldn't reach the Anthropic API. Check your internet connection.]")
        return

    print("\n\nSources:")
    for n, p in enumerate(result["passages"], 1):
        mark = "*" if n in result["cited"] else " "
        print(f" {mark}[{n}] {' > '.join([p['title'], *p['headings']])}")
        print(f"       {p['url']}")
    print(f"\n(* = cited. {result['input_tokens']:,} tokens in, {result['output_tokens']:,} out, "
          f"about ${result['cost_usd']:.3f}, model {result['model']})")


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the FastAPI + Supabase docs.")
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="passages to give Claude")
    parser.add_argument("--source", choices=["fastapi", "supabase"], help="use one source only")
    parser.add_argument("--balance", action="store_true",
                        help="always take passages from both sources (by default this happens "
                             "only when the question names both FastAPI and Supabase)")
    args = parser.parse_args()
    args.balance = True if args.balance else "auto"

    sys.stdout.reconfigure(encoding="utf-8")   # answers can contain emoji and arrows
    client = get_client()                      # fail fast if there's no API key
    print("Loading search index and model...")
    load_index()

    if args.question:
        print_answer(args.question, args, client)
        return

    print("Ask a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print_answer(question, args, client)


if __name__ == "__main__":
    main()
