"""
Inspect the Chroma vector store — a debugging/verification tool, not part
of the main pipeline. Use this any time you want to check what's actually
stored before trusting retrieval results.

Usage:
    python inspect_db.py                              # summary stats + sample
    python inspect_db.py --file cluster-networking.md  # all chunks from one file
    python inspect_db.py --search "pod networking"     # keyword search in stored text
"""

import sys

import chromadb

CHROMA_PERSIST_DIR = "./.chroma"
COLLECTION_NAME = "k8s-concepts"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


def show_summary(collection: chromadb.Collection) -> None:
    total = collection.count()
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks: {total}\n")

    sample = collection.peek(limit=3)
    print("Sample entries:")
    for i in range(len(sample["ids"])):
        print(f"\n  ID: {sample['ids'][i]}")
        print(f"  Metadata: {sample['metadatas'][i]}")
        print(f"  Text preview: {sample['documents'][i][:150]}...")


def show_by_file(collection: chromadb.Collection, filename: str) -> None:
    results = collection.get(
        where={"source_file": filename},
        include=["documents", "metadatas"],
    )
    print(f"Found {len(results['ids'])} chunks from '{filename}'\n")
    for i in range(len(results["ids"])):
        print(f"  [{results['ids'][i]}] section: {results['metadatas'][i]['section_header']}")
        print(f"  {results['documents'][i][:200]}...\n")


def keyword_search(collection: chromadb.Collection, keyword: str) -> None:
    """Plain substring search over stored text — not semantic, just for
    quickly checking whether specific content made it into the DB at all."""
    results = collection.get(include=["documents", "metadatas"])
    keyword_lower = keyword.lower()
    matches = [
        (results["ids"][i], results["documents"][i], results["metadatas"][i])
        for i in range(len(results["ids"]))
        if keyword_lower in results["documents"][i].lower()
    ]

    print(f"Found {len(matches)} chunks containing '{keyword}'\n")
    for chunk_id, text, meta in matches[:10]:
        print(f"  [{chunk_id}] {meta['doc_title']} > {meta['section_header']}")
        idx = text.lower().find(keyword_lower)
        snippet_start = max(0, idx - 60)
        print(f"  ...{text[snippet_start: idx + 100]}...\n")


if __name__ == "__main__":
    collection = get_collection()

    if "--file" in sys.argv:
        target = sys.argv[sys.argv.index("--file") + 1]
        show_by_file(collection, target)
    elif "--search" in sys.argv:
        term = sys.argv[sys.argv.index("--search") + 1]
        keyword_search(collection, term)
    else:
        show_summary(collection)