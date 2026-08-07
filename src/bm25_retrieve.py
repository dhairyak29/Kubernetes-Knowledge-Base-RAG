"""
Retrieval module — semantic search over the Chroma vector store.

This is Phase 2's retrieval step: a clean, reusable function that other
parts of the pipeline (generation, the CLI, later the FastAPI service)
can import directly, instead of re-writing query logic each time.

Note: this is semantic-only retrieval. Hybrid search (semantic + keyword)
and re-ranking come in Phase 3 — intentionally kept simple here so we can
get an end-to-end answer working first, then improve retrieval quality
once we can measure it.

Usage (standalone test):
    python retrieve.py "how does pod to pod communication work"
"""


from s5_retrieve import DEFAULT_TOP_K, RetrievedChunk, CHROMA_PERSIST_DIR, COLLECTION_NAME
import re
from rank_bm25 import BM25Okapi
import sys
import chromadb


# Cached at module level so the index is built once and reused across calls,
# same caching pattern used in retrieve.py for the embedding model/collection.
_bm25_index = None
_bm25_ids = None
_bm25_documents = None
_bm25_metadatas = None

def tokenize(text: str) -> list[str]:
    """
    Simple lowercase, alphanumeric tokenizer. BM25 works on exact term
    overlap, so no need for anything fancier than this for a first pass —
    stemming/stopword removal are reasonable future improvements to test.
    """

    return re.findall(r"[a-z0-9]+", text.lower())


def built_index():
    global _bm25_index, _bm25_ids, _bm25_documents, _bm25_metadatas

    client = chromadb.PersistentClient(CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_ids = all_data["ids"]
    _bm25_documents = all_data["documents"]
    _bm25_metadatas =  all_data["metadatas"]

    _tokenize_corpus = [tokenize(doc) for doc in _bm25_documents]
    _bm25_index = BM25Okapi(_tokenize_corpus)


def _get_index():
    if _bm25_index is None:
        built_index()
    return _bm25_index, _bm25_ids, _bm25_documents, _bm25_metadatas


def keyword_retrieve(query: str, top_k : int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Returns the top_k chunks ranked by BM25 keyword relevance score
    (higher score = more relevant, unlike cosine distance where lower is
    better — kept as plain dicts here since the scoring scale is different
    from semantic search; retrieve.py handles reconciling the two).
    """

    index, ids, documents, metadatas = _get_index()

    tokenized_query = tokenize(query)
    scores = index.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key= lambda i : scores[i], reverse=True)[:top_k]

    results = []

    for i in ranked_indices:
        results.append(
            {
                "chunk_id": ids[i],
                "text": documents[i],
                "doc_title": metadatas[i]["doc_title"],
                "source_file": metadatas[i]["source_file"],
                "section_header": metadatas[i]["section_header"],
                "bm25_score": scores[i],
            }
        )
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python s7_bm25_search.py "your question"')
        sys.exit(1)
 
    query = sys.argv[1]
    results = keyword_retrieve(query)
 
    print(f"Query: '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] bm25_score={r['bm25_score']:.2f}  {r['doc_title']} > {r['section_header']}")
        print(f"    {r['text'][:150].replace(chr(10), ' ')}...\n")

