"""
load_docs.py - Step 1 of the RAG pipeline: load, clean and save the raw docs.

Reads the FastAPI (.md) and Supabase (.mdx) documentation from data/, turns
each page into clean Markdown text, and writes data/processed/docs.jsonl with
one JSON object per line:

    {"id", "source", "section", "title", "description", "url", "text"}

Standard library only. Run with:  python load_docs.py

How it works (per page):
  1. Fill in missing content   (FastAPI code placeholders, Supabase partials)
  2. Pull out metadata         (frontmatter -> title / description)
  3. Protect code              (swap code blocks + `inline code` for tokens)
  4. Clean the markup          (boxes, components, HTML, images, links, ...)
  5. Restore code              (put the untouched code back)
Step 3 is what guarantees rule 10: the cleaning regexes simply can't see code.
"""

import hashlib
import html
import json
import posixpath
import re
import textwrap
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

# --------------------------------------------------------------------------
# Paths and settings
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FASTAPI_DOCS = ROOT / "data/fastapi/docs/en/docs"
FASTAPI_REPO = ROOT / "data/fastapi"
SUPABASE_CONTENT = ROOT / "data/supabase/apps/docs/content"
SUPABASE_PARTIALS = SUPABASE_CONTENT / "_partials"
OUTPUT = ROOT / "data/processed/docs.jsonl"

FASTAPI_URL = "https://fastapi.tiangolo.com"
SUPABASE_URL = "https://supabase.com"

MIN_CHARS = 100          # rule 16: drop pages shorter than this
MAX_PARTIAL_DEPTH = 5    # rule 2: partials inside partials, at most this deep

# Rule 15: FastAPI pages about the project itself, not about using FastAPI.
FASTAPI_SKIP = {
    "release-notes", "fastapi-people", "management", "management-tasks",
    "contributing", "translations", "external-links", "help-fastapi",
    "newsletter", "translation-banner",
}

# Rule 8: lowercase tags we treat as HTML. Anything else that starts with a
# lowercase letter (like <project-ref>) is left alone because it's a placeholder.
HTML_TAGS = (
    "a abbr b big blockquote br center code dd del details dfn div dl dt em "
    "figcaption figure font h1 h2 h3 h4 h5 h6 hr i iframe img input ins kbd "
    "label li mark nav ol p picture pre s section small source span strong "
    "sub summary sup table tbody td th thead tr u ul video"
).split()

# Every fix the cleaner makes is counted here, keyed by rule (for the report).
STATS = Counter()
# References (code files, partials) we couldn't find: (page id, reference).
MISSING = []

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def read(path):
    return path.read_text(encoding="utf-8")


def sub_count(pattern, repl, text, rule, flags=0):
    """re.sub that also adds the number of replacements to STATS[rule]."""
    new_text, n = re.subn(pattern, repl, text, flags=flags)
    STATS[rule] += n
    return new_text


def attr(attrs, name):
    """Read one attribute value (name="x" or name='x' or name={x}) from a tag."""
    m = re.search(rf'\b{name}\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|\{{([^{{}}]*)\}})', attrs)
    if not m:
        return None
    value = next(g for g in m.groups() if g is not None)
    return value.strip().strip("'\"")


def bold(label):
    """Wrap a label in **bold**, without producing **** or nested ** markers."""
    label = label.replace("**", "").strip()
    return f"**{label}**" if label else ""


# A JSX/HTML tag's attribute list. Attribute values can be "..", '..' or {..}
# with up to two levels of nested braces (e.g. src={{ light: '...' }}).
# Quoted strings inside braces may contain anything, even more braces:
#   variables={{ "framework": "{{ .framework }}" }}
# This lets us match multi-line tags without stopping at a '>' inside {...}.
_STR = r"\"[^\"]*\"|'[^']*'"
BRACES = (r"\{(?:" + _STR + r"|[^{}\"']|\{(?:" + _STR + r"|[^{}\"']|\{(?:" + _STR
          + r"|[^{}\"'])*\})*\})*\}")
ATTRS = (
    r"(?P<attrs>(?:\s+[\w:.$-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|" + BRACES
    + r"|[^\s\"'>/{}]+))?)*)\s*"
)


def tag_re(name):
    """Regex for an opening (or self-closing) tag called `name`."""
    return rf"<{name}{ATTRS}/?>"


# --------------------------------------------------------------------------
# Rule 3 / 10: protecting code
# --------------------------------------------------------------------------
# We swap every fenced block and `inline code` span for a token like
#  12  (private-use characters no regex below will touch).
# At the end, restore() puts the original code back.

TOKEN_RE = re.compile("(\\d+)")
# A ``` fence's info string (e.g. "python") can't contain backticks, so a line
# like ```const x = 1``` is inline code written with triple backticks, not
# the start of a code block (this is the CommonMark rule).
FENCE_OPEN = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}(?=[^`]*$)|~{3,})(?P<info>.*)$")


def strip_console_markup(code):
    """Rule 9: remove <font>/<span>/... colouring from ```console output."""
    new, n = re.subn(r"</?(?:font|span|u|b|i|strong|em)\b[^>]*>", "", code)
    STATS["9 console tags"] += n
    return html.unescape(new)


def protect(text):
    """Replace code with tokens. Returns (text_with_tokens, list_of_code)."""
    store = []

    def keep(code):
        store.append(code)
        return f"{len(store) - 1}"

    # Fenced code blocks: walk line by line so nesting/indentation is exact.
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = FENCE_OPEN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, fence = m["indent"], m["fence"]
        # Find the closing fence: same character, at least as long.
        close = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        block = lines[i:j + 1]
        # Remove the fence's own indentation (Supabase nests code inside
        # components). This is how Markdown reads it anyway, so the code
        # itself does not change.
        block = [ln[len(indent):] if ln.startswith(indent) else ln.lstrip() for ln in block]
        code = "\n".join(block)
        if m["info"].strip().lower() == "console":
            code = strip_console_markup(code)
        out.append(indent + keep(code))
        i = j + 1
    text = "\n".join(out)

    # Inline code: `x` or ``x`` on a single line.
    text = re.sub(r"(?<!`)(`+)(?!`)([^\n]+?)(?<!`)\1(?!`)", lambda m: keep(m[0]), text)
    return text, store


def restore(text, store):
    # Loop because a token's code could, in theory, hold another token.
    while TOKEN_RE.search(text):
        text = TOKEN_RE.sub(lambda m: store[int(m[1])], text)
    return text


# --------------------------------------------------------------------------
# Rule 4: frontmatter
# --------------------------------------------------------------------------

def split_frontmatter(text):
    """Split off a leading ---...--- block. Returns (frontmatter, body)."""
    m = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", text, flags=re.S)
    if not m:
        return "", text
    return m[1], text[m.end():]


def parse_yaml_value(raw):
    raw = raw.strip()
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        return raw[1:-1].replace("''", "'")            # YAML: '' is an escaped '
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return raw


def parse_frontmatter(fm):
    """Tiny parser for the bits we need. TOML (key = value) or simple YAML."""
    if not fm:
        return {}
    if re.search(r"^\s*title\s*=", fm, flags=re.M):     # troubleshooting = TOML
        try:
            return tomllib.loads(fm)
        except tomllib.TOMLDecodeError:
            pass  # fall through to the line-based parser below
    data, lines, i = {}, fm.split("\n"), 0
    while i < len(lines):
        m = re.match(r"^([\w-]+)\s*[:=]\s*(.*)$", lines[i])
        i += 1
        if not m:
            continue
        key, value = m[1], m[2]
        if value.strip() in (">", ">-", "|", "|-", ""):
            # Multi-line YAML value: take the following indented lines.
            parts = []
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            value = " ".join(p for p in parts if p)
        data[key] = parse_yaml_value(value)
    return data


# --------------------------------------------------------------------------
# Rule 1: FastAPI {* ../../docs_src/... *} code placeholders
# --------------------------------------------------------------------------

LANGS = {".py": "python", ".js": "javascript", ".html": "html", ".json": "json",
         ".toml": "toml", ".sh": "bash", ".txt": "text", ".css": "css"}


def parse_ranges(spec):
    """'1:11' -> [(1, 11)];  '14:15,17:18' -> [(14, 15), (17, 18)];  '3' -> [(3, 3)]"""
    ranges = []
    for part in spec.split(","):
        a, _, b = part.strip().partition(":")
        if a:
            ranges.append((int(a), int(b or a)))
    return ranges


def pick_lines(lines, ranges, comment):
    """Take 1-indexed inclusive ranges; mark skipped gaps with a '...' comment."""
    chunks = [lines[a - 1:b] for a, b in ranges]
    joined = []
    for k, chunk in enumerate(chunks):
        if k:
            joined.append(f"{comment} ...")
        joined.extend(chunk)
    return joined


def expand_fastapi_code(text, page_id):
    pasted = set()   # files already pasted in full on this page

    def replace(m):
        body = m[1]
        path_m = re.search(r"(?:\.\./)*(\S+\.\w+)", body)
        if not path_m:
            return m[0]
        # "../../docs_src/x.py" -> data/fastapi/docs_src/x.py. Almost all
        # placeholders point into docs_src/; one points into the fastapi
        # package source, which we didn't download.
        rel = path_m[1]
        src = FASTAPI_REPO / rel
        if not src.is_file():
            MISSING.append((page_id, rel))
            STATS["1 code placeholders (not found)"] += 1
            return f"[Code example not included: {Path(rel).name}]"
        lines = read(src).rstrip("\n").split("\n")
        ln = re.search(r"\bln\[([^\]]+)\]", body)
        hl = re.search(r"\bhl\[([^\]]+)\]", body)
        comment = "#" if src.suffix == ".py" else "//"

        if rel in pasted and hl:
            # Already shown above: just show the highlighted lines.
            code = pick_lines(lines, parse_ranges(hl[1]), comment)
            STATS["1 code placeholders (highlight only)"] += 1
        elif ln:
            code = pick_lines(lines, parse_ranges(ln[1]), comment)
            STATS["1 code placeholders (line range)"] += 1
        elif rel in pasted:
            STATS["1 code placeholders (repeat note)"] += 1
            return f"(Same code as above: {Path(rel).name})"
        else:
            code = lines
            pasted.add(rel)
            STATS["1 code placeholders (full file)"] += 1
        lang = LANGS.get(src.suffix, src.suffix.lstrip("."))
        return f"```{lang}\n" + "\n".join(code) + "\n```"

    return re.sub(r"\{\*(.*?)\*\}", replace, text)


# --------------------------------------------------------------------------
# Rules 2 + 3: Supabase <$Partial /> and <$CodeSample />
# --------------------------------------------------------------------------

def expand_partials(text, page_id, depth=0):
    def replace(m):
        attrs = m["attrs"]
        path = attr(attrs, "path")
        if not path:
            return ""
        src = SUPABASE_PARTIALS / path
        if depth >= MAX_PARTIAL_DEPTH:
            MISSING.append((page_id, f"_partials/{path} (too deep)"))
            return ""
        if not src.is_file():
            MISSING.append((page_id, f"_partials/{path}"))
            return ""
        # variables={{ "provider": "GitHub", "tab": "web" }}
        variables = dict(re.findall(r'"([\w-]+)"\s*:\s*"([^"]*)"', attrs))
        _, content = split_frontmatter(read(src))

        def fill(v):
            if v[1] in variables:
                STATS["2 partial variables filled"] += 1
                return variables[v[1]]
            STATS["2 partial variables left blank"] += 1
            return v[0]
        content = re.sub(r"\{\{\s*\.([\w-]+)\s*\}\}", fill, content)
        STATS["2 partials inserted"] += 1
        # The partial may include more partials: expand those too.
        return "\n" + expand_partials(content, page_id, depth + 1).strip("\n") + "\n"

    return re.sub(tag_re(r"\$Partial"), replace, text)


def replace_code_samples(text):
    def replace(m):
        attrs = m["attrs"]
        name = None
        meta = attr(attrs, "meta") or ""
        nm = re.search(r"name=(\S+)", meta)
        if nm:
            name = nm[1]
        elif attr(attrs, "path"):
            name = Path(attr(attrs, "path")).name
        STATS["3 code samples noted"] += 1
        return f"[Code sample not included: {name or 'unknown'}]"

    return re.sub(tag_re(r"\$CodeSample"), replace, text)


# --------------------------------------------------------------------------
# Rule 6: FastAPI /// admonition boxes
# --------------------------------------------------------------------------

def clean_fastapi_boxes(text):
    def replace(m):
        kind, title = m["kind"], m["title"]
        if title and title.strip():
            label = bold(title)                    # "/// note | Some Title"
        elif kind:
            label = bold(kind.capitalize() + ":")  # "/// tip" -> **Tip:**
        else:
            label = ""                             # closing "///"
        STATS["6 box markers"] += 1
        return m["indent"] + label

    return re.sub(
        r"^(?P<indent>[ \t]*)/{3,4}[ \t]*(?P<kind>[A-Za-z]+)?[ \t]*(?:\|(?P<title>[^\n]*))?[ \t]*$",
        replace, text, flags=re.M)


# --------------------------------------------------------------------------
# Rule 7: Supabase / JSX components
# --------------------------------------------------------------------------

def clean_components(text):
    # Components that carry a useful label: turn the label into bold text.
    def label_from(*names, default=None):
        def replace(m):
            STATS["7 component tags"] += 1
            for name in names:
                value = attr(m["attrs"], name)
                if value:
                    return "\n" + bold(value) + "\n"
            if default:
                kind = attr(m["attrs"], "type") or default
                return "\n" + bold(kind.capitalize() + ":") + "\n"
            return ""
        return replace

    text = re.sub(tag_re("TabPanel"), label_from("label"), text)
    text = re.sub(tag_re("Admonition"), label_from("title", default="note"), text)
    text = re.sub(tag_re("AccordionItem"), label_from("header"), text)
    text = re.sub(tag_re(r"StepHikeCompact\.Details"), label_from("title"), text)
    text = re.sub(tag_re("GlassPanel"), label_from("title"), text)

    # Small inline components that stand for a value.
    text = sub_count(tag_re("Price"), lambda m: f"${attr(m['attrs'], 'price')}", text, "7 component tags")
    text = sub_count(tag_re("IconCheck"), "Yes", text, "7 component tags")
    text = sub_count(tag_re("IconX"), "No", text, "7 component tags")
    text = sub_count(tag_re("YouTube"),
                     lambda m: f"[Video: {attr(m['attrs'], 'title') or 'YouTube'}]",
                     text, "7 component tags")

    # import/export lines (MDX module code) and {/* comments */}.
    text = sub_count(r"^(?:import|export)\s[^\n]*(?:\n[ \t]+[^\n]*)*$\n?", "", text, "7 import/export lines", re.M)
    text = sub_count(r"\{/\*.*?\*/\}", "", text, "7 MDX comments", re.S)
    text = sub_count(r"\{\s*(['\"])\s*\1\s*\}", " ", text, "7 component tags")   # {' '}

    # Everything else: drop the tag, keep what's inside. Component names
    # start with $ or a capital and contain a lowercase letter, so ALL_CAPS
    # placeholders like <SUPABASE_URL> are left alone.
    name = r"(?:\$[\w.]+|[A-Z][\w.]*[a-z][\w.]*)"
    text = sub_count(rf"</{name}\s*>", "", text, "7 component tags")
    text = sub_count(rf"<{name}{ATTRS}/?>", "", text, "7 component tags")
    return text


# --------------------------------------------------------------------------
# Rules 8, 12: HTML tags and images
# --------------------------------------------------------------------------

def clean_images(text):
    def md_image(m):
        STATS["12 images"] += 1
        alt = m[1].strip()
        return f"[Image: {alt}]" if alt else "[Image]"

    def tag_image(m):
        STATS["12 images"] += 1
        alt = attr(m["attrs"], "alt")
        return f"[Image: {alt}]" if alt and "{{" not in alt else "[Image]"

    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", md_image, text)       # ![alt](src)
    text = re.sub(tag_re("Image"), tag_image, text)                 # <Image .../>
    text = re.sub(tag_re("img"), tag_image, text)                   # <img ...>
    return text


def clean_html(text):
    # Remove whole elements whose content is not prose.
    text = sub_count(r"<(style|script)\b.*?</\1>", "", text, "8 html tags", re.S | re.I)
    text = sub_count(r"<!--.*?-->", "", text, "8 html comments", re.S)
    text = sub_count(r"<(iframe|video)\b.*?(?:</\1>|/>)", "", text, "8 html tags", re.S | re.I)

    # <abbr title="Software Development Kits">SDKs</abbr> -> SDKs (Software Development Kits)
    # The title is the definition, which is exactly what a reader would ask about.
    def define(m):
        STATS["8 html tags"] += 1
        title = attr(m["attrs"], "title")
        return f"{m['body']} ({title})" if title else m["body"]
    text = re.sub(r"<(abbr|dfn)" + ATTRS + r">(?P<body>.*?)</\1>", define, text, flags=re.S)

    # <a href="x">text</a> -> [text](x), so rule 14 can make it absolute.
    def anchor(m):
        STATS["8 html tags"] += 1
        href = attr(m["attrs"], "href")
        body = m["body"].strip()
        return f"[{body}]({href})" if href and body and "{{" not in href else body
    text = re.sub(r"<a" + ATTRS + r">(?P<body>.*?)</a>", anchor, text, flags=re.S)

    # <code>x</code> -> `x`
    text = sub_count(r"<code>(.*?)</code>", r"`\1`", text, "8 html tags", re.S)
    text = sub_count(r"<br\s*/?>", " ", text, "8 html tags")

    # Any other known HTML tag: drop the tag, keep the words.
    names = "|".join(HTML_TAGS)
    text = sub_count(rf"</?(?:{names})\b{ATTRS}/?>", "", text, "8 html tags", re.I)
    return text


# --------------------------------------------------------------------------
# Rule 14: absolute links
# --------------------------------------------------------------------------

def fastapi_path_to_url(rel):
    """tutorial/first-steps.md -> https://fastapi.tiangolo.com/tutorial/first-steps/"""
    rel = rel.replace("\\", "/")
    if rel.endswith(".md"):
        rel = rel[:-3]
        if rel == "index":
            rel = ""
        elif rel.endswith("/index"):
            rel = rel[:-6]
        return f"{FASTAPI_URL}/{rel}/" if rel else f"{FASTAPI_URL}/"
    return f"{FASTAPI_URL}/{rel}"


def make_link_absolute(link, source, page_rel, page_url):
    if not link or link.startswith(("#", "http://", "https://", "mailto:", "")):
        return link
    if link.startswith("/"):
        return (SUPABASE_URL if source == "supabase" else FASTAPI_URL) + link
    if source == "fastapi":
        # Relative to the .md file's folder, like MkDocs does.
        path, _, anchor = link.partition("#")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page_rel), path))
        url = fastapi_path_to_url(target)
        return url + ("#" + anchor if anchor else "")
    return urljoin(page_url, link)


def fix_links(text, source, page_rel, page_url):
    def inline(m):
        new = make_link_absolute(m["url"], source, page_rel, page_url)
        if new != m["url"]:
            STATS["14 links made absolute"] += 1
        return f"[{m['text']}]({new}{m['title'] or ''})"

    def reference(m):
        new = make_link_absolute(m[2], source, page_rel, page_url)
        if new != m[2]:
            STATS["14 links made absolute"] += 1
        return f"{m[1]}{new}"

    text = re.sub(r"\[(?P<text>[^\]\n]*)\]\((?P<url>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)", inline, text)
    text = re.sub(r"^(\[[^\]]+\]:\s*)(\S+)", reference, text, flags=re.M)
    return text


# --------------------------------------------------------------------------
# Whitespace tidy-up
# --------------------------------------------------------------------------

def tidy(text, source):
    # Supabase nests prose inside components, so after removing the tags the
    # prose is still indented (which Markdown would read as a code block).
    # Dedent each paragraph by its own common indent; relative indentation
    # inside a paragraph (nested lists) is kept.
    if source == "supabase":
        text = "\n\n".join(textwrap.dedent(p) for p in re.split(r"\n[ \t]*\n", text))
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)     # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)               # 3+ blank lines -> 1
    return text.strip()


# --------------------------------------------------------------------------
# The cleaning pipeline for one page
# --------------------------------------------------------------------------

def clean_page(raw, source, page_rel, page_id, page_url):
    """Returns (frontmatter_dict, clean_text)."""
    raw = raw.replace("\r\n", "\n")

    # 1) Fill in missing content (before protecting, so inserted code is
    #    protected too).
    fm_raw, body = split_frontmatter(raw)
    if source == "fastapi":
        body = expand_fastapi_code(body, page_id)
    else:
        body = expand_partials(body, page_id)
        body = replace_code_samples(body)

    # 2) Metadata (rule 4).
    meta = parse_frontmatter(fm_raw)
    if fm_raw:
        STATS["4 frontmatter blocks"] += 1

    # 3) Protect code.
    text, code = protect(body)

    # 4) Clean markup (outside code only).
    if source == "fastapi":
        text = sub_count(r"[ \t]*\{\s*#[\w.-]+\s*\}[ \t]*$", "", text, "5 heading anchors", re.M)
        text = clean_fastapi_boxes(text)
    text = clean_images(text)                   # before components/HTML strip
    if source == "supabase":
        text = clean_components(text)
    text = clean_html(text)
    text = sub_count(r"\{%.*?%\}", "", text, "13 template code", re.S)
    text = sub_count(r"\{\{.*?\}\}", "", text, "13 template code", re.S)
    unescaped = html.unescape(text)
    STATS["11 html entities"] += len(re.findall(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", text, re.I))
    text = fix_links(unescaped, source, page_rel, page_url)
    text = tidy(text, source)

    # 5) Put the code back.
    return meta, restore(text, code)


# --------------------------------------------------------------------------
# Finding the pages
# --------------------------------------------------------------------------

def fastapi_pages():
    skipped = Counter()
    for path in sorted(FASTAPI_DOCS.rglob("*.md")):
        rel = path.relative_to(FASTAPI_DOCS).as_posix()
        if path.stem in FASTAPI_SKIP:
            skipped["project page"] += 1
        elif rel.startswith("reference/"):
            skipped["reference/ folder"] += 1
        elif path.name.startswith("_"):
            skipped["starts with _"] += 1
        else:
            yield path, rel
    STATS_SKIPPED["fastapi"] = skipped


def supabase_pages():
    skipped = Counter()
    for folder in ("guides", "troubleshooting"):
        for path in sorted((SUPABASE_CONTENT / folder).rglob("*.mdx")):
            rel = path.relative_to(SUPABASE_CONTENT).as_posix()
            if path.name.startswith("_"):          # e.g. troubleshooting/_template.mdx
                skipped["starts with _"] += 1
            else:
                yield path, rel
    STATS_SKIPPED["supabase"] = skipped


STATS_SKIPPED = {}


def supabase_url(rel):
    stem = rel[:-4]                                      # drop .mdx
    if stem.startswith("troubleshooting/"):
        return f"{SUPABASE_URL}/docs/guides/troubleshooting/{Path(stem).name}"
    if stem.endswith("/index"):
        stem = stem[:-6]
    return f"{SUPABASE_URL}/docs/{stem}"


def build_record(source, path, rel):
    stem_rel = rel.rsplit(".", 1)[0]
    page_id = f"{source}:{stem_rel}"
    parts = stem_rel.split("/")

    if source == "fastapi":
        url = fastapi_path_to_url(rel)
        section = parts[0] if len(parts) > 1 else "root"
    else:
        url = supabase_url(rel)
        if parts[0] == "troubleshooting":
            section = "troubleshooting"
        else:   # guides/<section>/...  (or guides/<section>.mdx)
            section = parts[1] if len(parts) > 1 else "guides"

    meta, text = clean_page(read(path), source, rel, page_id, url)

    title = str(meta.get("title") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", text, flags=re.M)
        title = m[1].strip() if m else Path(rel).stem.replace("-", " ").title()
    return {
        "id": page_id,
        "source": source,
        "section": section,
        "title": title,
        "description": str(meta.get("description") or "").strip(),
        "url": url,
        "text": text,
    }


# --------------------------------------------------------------------------
# Leftover check (for the report)
# --------------------------------------------------------------------------

def outside_code(text):
    """The text with fenced and inline code removed, for leftover checks."""
    prose, _ = protect(text)
    return TOKEN_RE.sub("CODE", prose)


def main():
    records, seen_hashes = [], set()
    dropped = Counter()

    for source, pages in (("fastapi", fastapi_pages()), ("supabase", supabase_pages())):
        for path, rel in pages:
            rec = build_record(source, path, rel)
            if len(rec["text"]) < MIN_CHARS:
                dropped[f"{source}: under {MIN_CHARS} chars"] += 1
                continue
            digest = hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                dropped[f"{source}: duplicate"] += 1
                continue
            seen_hashes.add(digest)
            records.append(rec)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- Report ----------------
    print(f"Saved {len(records)} pages to {OUTPUT.relative_to(ROOT)}\n")
    print("Pages per source:")
    for source in ("fastapi", "supabase"):
        recs = [r for r in records if r["source"] == source]
        chars = sum(len(r["text"]) for r in recs)
        skipped = dict(STATS_SKIPPED.get(source, {}))
        print(f"  {source:9} {len(recs):5} pages  {chars:>11,} chars   skipped: {skipped}")
    print(f"  {'total':9} {len(records):5} pages  {sum(len(r['text']) for r in records):>11,} chars")
    print(f"  dropped after cleaning: {dict(dropped)}")

    print("\nFixes per cleaning rule:")
    for rule in sorted(STATS, key=lambda k: (int(k.split()[0]), k)):
        print(f"  rule {rule:45} {STATS[rule]:6}")

    all_text = "\n".join(r["text"] for r in records)
    print("\nLeftover check (should be 0):")
    print(f"  '{{* '       {all_text.count('{* ')}")
    print(f"  '<$Partial'  {all_text.count('<$Partial')}")

    prose = "\n".join(outside_code(r["text"]) for r in records)
    print("\nLeftover markup outside code (review these):")
    for label, pattern in [
        ("component tags", r"</?(?:\$[\w.]+|[A-Z][\w.]*[a-z][\w.]*)[\s/>]"),
        ("html tags", r"</?(?:" + "|".join(HTML_TAGS) + r")\b[^>]*>"),
        ("****", r"\*\*\*\*"),
        ("{{", r"\{\{"),
        ("///", r"^\s*///"),
    ]:
        print(f"  {label:15} {len(re.findall(pattern, prose, flags=re.M))}")

    print(f"\nReferences not found ({len(MISSING)}):")
    for page_id, ref in MISSING:
        print(f"  {page_id}: {ref}")

    print("\n5 longest pages:")
    for r in sorted(records, key=lambda r: len(r["text"]), reverse=True)[:5]:
        print(f"  {len(r['text']):>8,}  {r['id']}")


if __name__ == "__main__":
    main()
