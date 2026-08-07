"""
Hybrid search — combines semantic search (s5_retrieve.py) and keyword
search (s7_bm25_search.py) using Reciprocal Rank Fusion (RRF).
 
This file is intentionally a separate layer on top of both retrieval
methods, rather than being folded into either one — s5 stays a clean
semantic-only baseline, s7 stays a clean BM25-only baseline, and this
file is the only place that knows how to merge them. That separation
makes it straightforward to compare semantic-only vs hybrid results
directly for the Phase 4 evaluation.
 
Usage (standalone test):
    python s8_hybrid_search.py "how does a service work"
"""


import sys
 
from s5_retrieve import retrieve, RetrievedChunk, DEFAULT_TOP_K
from bm25_retrieve import keyword_retrieve
 
RRF_K = 60           # standard damping constant for Reciprocal Rank Fusion
CANDIDATE_POOL = 20  # how many results to pull from EACH method before merging


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Runs semantic search and BM25 search independently, then merges their
    rankings with RRF. Using rank position (1st, 2nd, 3rd...) rather than
    raw scores avoids the problem that cosine distance and BM25 score are
    on completely different, incompatible scales.
    """
    semantic_results = retrieve(query, top_k=CANDIDATE_POOL)
    keyword_results = keyword_retrieve(query, top_k=CANDIDATE_POOL)


    # rrf_scores maps chunk_id -> combined score across both ranked lists
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}


    for rank, chunk in enumerate(semantic_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0/(RRF_K + rank)
        chunk_lookup[chunk.chunk_id] = {
                        "text": chunk.text,
            "doc_title": chunk.doc_title,
            "source_file": chunk.source_file,
            "section_header": chunk.section_header,
            "distance": chunk.distance,
        }

    for rank, result in enumerate(keyword_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        if cid not in chunk_lookup:
            # found by BM25 but not semantic search — no cosine distance exists
            chunk_lookup[cid] = {
                "text": result["text"],
                "doc_title": result["doc_title"],
                "source_file": result["source_file"],
                "section_header": result["section_header"],
                "distance": None,
            }

    ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    merged = []
    for cid in ranked_ids:
        info = chunk_lookup[cid]
        merged.append(
            RetrievedChunk(
                chunk_id=cid,
                text=info["text"],
                doc_title=info["doc_title"],
                source_file=info["source_file"],
                section_header=info["section_header"],
                # Chunks found only via BM25 have no real cosine distance.
                # We substitute a value derived from the RRF score (inverted
                # so lower still reads as "better") purely for consistent
                # display — this is NOT a true cosine distance and shouldn't
                # be treated as comparable to one in any eval numbers.
                distance=info["distance"] if info["distance"] is not None else 1.0 - rrf_scores[cid],
            )
        )
 
    return merged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python s8_hybrid_search.py "your question"')
        sys.exit(1)
 
    query = sys.argv[1]
    results = hybrid_search(query)
 
    print(f"[Hybrid] Query: '{query}'\n")
    for i, c in enumerate(results, 1):
        print(f"[{i}] distance={c.distance:.4f}  {c.doc_title} > {c.section_header}")
        print(f"    {c.text[:150].replace(chr(10), ' ')}...\n")