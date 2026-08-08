"""
Step 2: Clean
Takes the raw documents loaded by load_docs.py and strips out Hugo-specific
syntax (front matter, shortcodes) that would otherwise pollute embeddings
and confuse the LLM at generation time.
 
Usage:
    python clean_docs.py /path/to/concepts
"""

import re
import sys
from pathlib import Path
from s1_load_docs import load_markdown_docs

def extract_front_matter(raw_text: str) -> tuple[dict, str]:
        """
    Splits a markdown file into (front_matter_dict, body_text).
 
    Hugo front matter looks like:
        ---
        title: "Pods"
        weight: 10
        ---
        # actual content starts here
 
    We parse just 'title' since that's the most useful bit of metadata —
    good enough for citations without needing a full YAML parser.
    """
        front_matter = {}
        body = raw_text

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", raw_text, re.DOTALL)

        if match:
                fm_block, body = match.group(1), match.group(2)
                title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm_block, re.MULTILINE)
                if title_match:
                    front_matter["title"] = title_match.group(1)

        return front_matter, body


def strip_shortcodes(text: str) -> str:
    """
    Removes Hugo shortcodes while preserving any human-readable text they carry.
 
    Two cases:
    1. Inline shortcodes with a text="..." attribute, e.g.
       {{< glossary_tooltip text="workload" term_id="workload" >}}
       -> keep "workload", drop the rest of the tag.
    2. Wrapper shortcodes with opening/closing tags, e.g.
       {{< note >}} some warning text {{< /note >}}
       -> keep "some warning text", drop the tags.
    """
    # Case 1: pull out text="..." before removing the tag
    def replace_with_text_attr(m: re.Match) -> str:
        return m.group(1)
 
    text = re.sub(
        r'\{\{[<%]\s*\w+[^%>]*?text=["\']([^"\']+)["\'][^%>]*?[%>]\}\}',
        replace_with_text_attr,
        text,
    )
 
    # Case 2: remove HTML comments entirely — covers both short Hugo section
    # markers like <!-- overview --> AND long embedded content like Mermaid
    # diagram links (<!-- https://mermaid-js.github.io/...#pako:xyz... -->),
    # which can span hundreds of characters and pollute chunking/BM25 if left in.
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    
    # Case 2b: known heading shortcodes -> replace with readable text so we
    # don't leave an empty "## " header after stripping.
    known_headings = {
        "whatsnext": "What's next",
        "whatnext": "What's next",
        "prerequisites": "Prerequisites",
    }
    def replace_heading(m: re.Match) -> str:
        key = m.group(1).strip().strip('"').lower()
        return known_headings.get(key, key.replace("-", " ").title())
 
    text = re.sub(r'\{\{%\s*heading\s+"?([\w-]+)"?\s*%\}\}', replace_heading, text)
 
    # Case 3: strip remaining shortcode tags (opening and closing), keep content between them
    text = re.sub(r"\{\{[<%]\s*/?\s*[\w-]+[^%>]*?[%>]\}\}", "", text)
 
    return text
 
 
def strip_markdown_links_keep_text(text: str) -> str:
    """
    Converts [link text](url) -> link text.
    Keeps the readable text, drops the URL (not useful for embeddings).
    """
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
 
 
def normalize_whitespace(text: str) -> str:
    """Collapses excessive blank lines and trailing whitespace."""
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces per line
    text = re.sub(r"\n{3,}", "\n\n", text)           # 3+ blank lines -> 1
    return text.strip()


def clean_document(doc: dict) -> dict:
    """
    Runs the full cleaning pipeline on a single loaded document.
    Returns a new dict with 'title', 'clean_text', plus original metadata.
    """
    front_matter, body = extract_front_matter(doc["raw_text"])
    body = strip_shortcodes(body)
    body = strip_markdown_links_keep_text(body)
    body = normalize_whitespace(body)

    return {
        "path": doc["path"],
        "filename": doc["filename"],
        "title": front_matter.get("title", doc["filename"]),
        "clean_text": body,
    }

def clean_all(docs: list[dict]) -> list[dict]:
    return [clean_document(d) for d in docs]
 
 
def print_summary(raw_docs: list[dict], cleaned_docs: list[dict]) -> None:
    print(f"Cleaned {len(cleaned_docs)} documents\n")
    print("Before/after comparison (first doc):")
    if cleaned_docs:
        raw = raw_docs[15]
        clean = cleaned_docs[15]
        print(f"  Title: {clean['title']}")
        print(f"  Filename: {clean['filename']}")
        print(f"\n  --- BEFORE (raw, first 400 chars) ---\n{raw['raw_text'][:1000]}")
        print(f"\n  --- AFTER (cleaned, first 400 chars) ---\n{clean['clean_text'][:1000]}\n")

 
if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:
        print("Usage: python clean_docs.py /path/to/concepts")
        sys.exit(1)
 
    folder = sys.argv[1]
    raw_docs = load_markdown_docs(folder)
    cleaned = clean_all(raw_docs)
    print_summary(raw_docs, cleaned)