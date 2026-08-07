"""
Step 1: Load
Walks the Kubernetes docs 'concepts' folder and loads every markdown file
into memory as a list of documents with basic metadata.
 
Usage:
    python load_docs.py /path/to/website/content/en/docs/concepts
"""

import sys
from pathlib import Path

def load_markdown_docs(root_dir : str) -> list[dict]:
    """
    Recursively finds all .md files under root_dir and loads their raw content.

    Returns a list of dicts, each representing one document:
        {
            "path": str,        # full file path, used later for citations
            "filename": str,    # just the filename, e.g. "overview.md"
            "raw_text": str,    # untouched file content (front matter + markdown)
        }
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist {root_dir}")

    docs = []

    md_files = sorted(root.rglob("*.md"))

    for file_path in md_files:
        try:
            raw_text = file_path.read_text(encoding="utf-8")
            if(file_path.name == "networking.md"):
                print(raw_text)
        except UnicodeDecodeError:
            print(f"  !  Skipping {file_path} - encoding issue")
            continue

        docs.append({
            "path":str(file_path),
            "filename": file_path.name,
            "raw_text": raw_text
        })

    return docs

def print_summary(docs: list[dict]) -> None:
    total_chars = sum(len(d["raw_text"]) for d in docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Total characters: {total_chars:,}")
    print(f"Average doc length: {total_chars // max(len(docs), 1):,} characters")
    print("\nSample of loaded files:")
    for d in docs[:5]:
        print(f"  - {d['filename']} ({len(d['raw_text'])} chars)")


if __name__ == "__main__":
    print(sys.argv)
    if(len(sys.argv) != 2):
        print("Usage: python load_docs.py /path/to/concepts")
        sys.exit(1)

    folder = sys.argv[1]
    documents = load_markdown_docs(folder)
    print_summary(documents)
