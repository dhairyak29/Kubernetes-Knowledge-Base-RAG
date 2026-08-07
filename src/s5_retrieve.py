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

import sys
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PERSIST_DIR = "./.chroma"
COLLECTION_NAME = "k8s-concepts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# Module-level cache so we don't reload the model / reconnect to Chroma
# on every call — important once this is used inside a live API.
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_title: str
    source_file: str
    section_header: str
    distance: float  # lower = more similar (cosine distance)


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Embeds the query and returns the top_k most semantically similar chunks
    from the vector store, ranked by relevance (best first).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        retrieved.append(
            RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                doc_title=meta["doc_title"],
                source_file=meta["source_file"],
                section_header=meta["section_header"],
                distance=results["distances"][0][i],
            )
        )

    return retrieved


def format_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks into a single string ready to insert into an
    LLM prompt, with source labels so the model (and later, you) can trace
    which chunk supported which part of an answer.
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Source {i}: {c.doc_title} > {c.section_header}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    results = retrieve(query)

    print(f"Query: '{query}'\n")
    for i, c in enumerate(results, 1):
        print(f"[{i}] distance={c.distance:.4f}  {c.doc_title} > {c.section_header}")
        print(f"    {c.text[:150].replace(chr(10), ' ')}...\n")