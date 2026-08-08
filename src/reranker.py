"""
Re-ranking — takes the candidate pool from hybrid search and re-scores each
one with a cross-encoder, which judges query+chunk relevance jointly rather
than comparing separately-computed representations. More accurate than
either semantic or keyword search alone, but too expensive to run against
the whole corpus — so it only runs on the narrowed candidate pool.
 
Uses sentence-transformers' CrossEncoder class (no new dependency needed,
already included in the sentence-transformers package).
 
Usage (standalone test):
    python s9_rerank.py "how does a service work"
"""

import sys
 
from sentence_transformers import CrossEncoder
 
from src.s5_retrieve import RetrievedChunk, DEFAULT_TOP_K
from src.hybrid_search import hybrid_search, CANDIDATE_POOL
 
# A small, fast cross-encoder trained specifically for search relevance
# ranking (MS MARCO passage ranking dataset) — good accuracy/speed tradeoff
# for a portfolio project, runs fine on CPU.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
_reranker = None
 
 
def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker



def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Scores each candidate chunk against the query using a cross-encoder,
    then returns the top_k re-sorted by that score (highest relevance first).
    """
    if not candidates:
        return []
 
    model = _get_reranker()
 
    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
 
    # Pair each candidate with its cross-encoder score and sort descending
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
 
    reranked = []
    for chunk, score in scored[:top_k]:
        # Overwrite distance with the cross-encoder score, inverted so lower
        # still means "better" — keeps the RetrievedChunk display convention
        # consistent, but note this is a DIFFERENT scale than cosine distance
        # or the RRF-derived value from hybrid_search. Don't compare these
        # numbers directly across methods.
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_title=chunk.doc_title,
                source_file=chunk.source_file,
                section_header=chunk.section_header,
                distance=-float(score),  # negative so "lower = better" still holds visually
            )
        )
 
    return reranked
 
 
def retrieve_and_rerank(query: str, top_k: int = DEFAULT_TOP_K, candidate_pool: int = CANDIDATE_POOL) -> list[RetrievedChunk]:
    """
    Full pipeline: hybrid search for a broad candidate pool, then re-rank
    those candidates precisely and return the final top_k. This is the
    function generate.py will eventually call.
    """
    candidates = hybrid_search(query, top_k=candidate_pool)
    return rerank(query, candidates, top_k=top_k)



def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Scores each candidate chunk against the query using a cross-encoder,
    then returns the top_k re-sorted by that score (highest relevance first).
    """
    if not candidates:
        return []
 
    model = _get_reranker()
 
    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
 
    # Pair each candidate with its cross-encoder score and sort descending
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
 
    reranked = []
    for chunk, score in scored[:top_k]:
        # Overwrite distance with the cross-encoder score, inverted so lower
        # still means "better" — keeps the RetrievedChunk display convention
        # consistent, but note this is a DIFFERENT scale than cosine distance
        # or the RRF-derived value from hybrid_search. Don't compare these
        # numbers directly across methods.
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_title=chunk.doc_title,
                source_file=chunk.source_file,
                section_header=chunk.section_header,
                distance=-float(score),  # negative so "lower = better" still holds visually
            )
        )
 
    return reranked
 
 
def retrieve_and_rerank(query: str, top_k: int = DEFAULT_TOP_K, candidate_pool: int = CANDIDATE_POOL) -> list[RetrievedChunk]:
    """
    Full pipeline: hybrid search for a broad candidate pool, then re-rank
    those candidates precisely and return the final top_k. This is the
    function generate.py will eventually call.
    """
    candidates = hybrid_search(query, top_k=candidate_pool)
    return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python s9_rerank.py "your question"')
        sys.exit(1)
 
    query = sys.argv[1]
    results = retrieve_and_rerank(query)
 
    print(f"[Hybrid + Reranked] Query: '{query}'\n")
    for i, c in enumerate(results, 1):
        print(f"[{i}] cross_encoder_score={-c.distance:.4f}  {c.doc_title} > {c.section_header}")
        print(f"    {c.text[:150].replace(chr(10), ' ')}...\n")