# Kubernetes-Knowledge-Base-RAG — production RAG system with hybrid search and re-ranking

A retrieval-augmented Q&A system over the Kubernetes `/concepts/` documentation, built with an
open-weight LLM (Llama 3.3 via Groq) and a fully local, free embedding/retrieval stack. Answers
are grounded strictly in retrieved documentation — the system is instructed to say "I don't know"
rather than fall back on the model's own training knowledge, and this is verified through testing.

**Status: Phases 1-3 complete** (data pipeline, basic RAG, hybrid search + re-ranking).
Phase 4 (evaluation harness) is next.

## Why this exists

Generic LLMs either don't know about specific, fast-changing documentation (like a particular
version of Kubernetes' concepts) or blend real facts with plausible-sounding guesses. This project
answers questions about Kubernetes concepts using only the official docs as source material, with
visible citations back to the exact document and section used — so answers are checkable, not just
plausible.

## Architecture

```
Raw docs (Kubernetes /concepts/, 176 markdown files)
        │
        ▼
   1. Load      — walk directory, read every .md file
        │
        ▼
   2. Clean     — strip Hugo front matter, shortcodes, HTML comments, markdown links
        │
        ▼
   3. Chunk     — split by markdown headers, cap size to the embedding model's real
        │          token limit, with a hard token-window fallback for oversized sections
        ▼
   4. Embed     — sentence-transformers (all-MiniLM-L6-v2), stored in Chroma (local, persistent)
        │
        ▼
   5. Retrieve  — semantic search (Chroma) + BM25 keyword search (rank-bm25),
        │          merged via Reciprocal Rank Fusion
        ▼
   6. Re-rank   — cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores the merged
        │          candidate pool for final relevance ranking
        ▼
   7. Generate  — top chunks + question sent to Groq (Llama 3.3 70B), with a system
                   prompt enforcing "answer only from provided sources"
```

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM (generation) | Groq — `llama-3.3-70b-versatile` | Free tier, fast inference, OpenAI-compatible API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, no API cost or latency dependency |
| Vector DB | Chroma (local, persistent) | Zero setup, no account needed |
| Keyword search | `rank-bm25` | Standard BM25 implementation |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small, fast, purpose-built for relevance ranking |
| Package management | `uv` | Fast installs, lockfile-based reproducibility |

No paid API keys are required anywhere in this stack.

## Project structure

```
src/
├── s1_load_docs.py      # walks the docs folder, reads raw markdown
├── s2_clean_docs.py      # strips Hugo front matter, shortcodes, HTML comments
├── s3_chunk_docs.py       # markdown-aware chunking with a hard token-cap fallback
├── s4_embed_docs.py        # embeds chunks, stores them in Chroma
├── s5_retrieve.py            # semantic-only retrieval (clean baseline, unmodified)
├── s6_generate.py              # retrieval + Groq generation, with citation formatting
├── bm25_search.py             # keyword search over the same chunk set
├── hybrid_search.py             # merges semantic + BM25 via Reciprocal Rank Fusion
├── rerank.py                      # cross-encoder re-ranking of hybrid candidates
├── cli.py                              # interactive chat loop for manual testing
└── inspect_db.py                        # utility for browsing/debugging the vector store
```

## Results: retrieval quality by stage

Benchmarked on the query **"how does a service work"**, chosen because it exposed real, distinct
failure modes at each stage — not a cherry-picked easy case.

| Stage | Top result | Verdict |
|---|---|---|
| Semantic-only | Cloud Controller Manager > Service controller | Tangential — related infra, not the concept itself |
| Hybrid (semantic + BM25) | Service Accounts > Introduction | Wrong concept — BM25 noise from generic word overlap ("work") pulled in unrelated chunks |
| **Hybrid + re-ranked** | **Service > Defining a Service** | Correct — cross-encoder correctly distinguished "Service" from "Service Accounts" and demoted BM25 noise |

Formal metrics (faithfulness, context precision/recall via RAGAS, across a full test set) are
being built in Phase 4 — the table above is a single representative example, not the full
evaluation.

## Real bugs found and fixed along the way

Documented here because catching these mattered more than any single design choice:

1. **Silent word loss during cleaning** — a shortcode-stripping regex was deleting real words
   (e.g. `{{< glossary_tooltip text="workload" ... >}}` → the word "workload" vanished along with
   the tag). Fixed by extracting `text="..."` attributes before removing the tag.
2. **Chunks silently exceeding the embedding model's token limit** — chunks were capped against
   the LLM's context window, not the embedding model's actual (much smaller) limit, so some chunks
   were getting silently truncated during embedding. Re-sized the cap to match the embedding
   model, and added a hard token-window fallback for content (bullet lists, code blocks) that
   defeats sentence-based splitting.
3. **Duplicate chunk IDs causing a hard crash** — Hugo reuses filenames like `_index.md` across
   many subfolders; using filename alone as the chunk ID caused collisions. Fixed by keying off
   the full file path instead.
4. **Corrupted content polluting BM25 rankings** — some doc pages embed Mermaid diagrams as
   long HTML comments containing base64-encoded data; the comment-stripping regex only matched
   short single-word comments, so this encoded gibberish leaked into chunks and dominated BM25
   scores (rare tokens score very high under BM25's IDF weighting). Fixed by matching HTML
   comments of any length/content.

## Getting started

```powershell
git clone <this-repo>
cd RAG-Kubernetes-Docs
uv venv --python 3.11
.venv\Scripts\activate
uv sync
```

Add a `.env` file with your free Groq API key ([console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=gsk_...
```

Build the pipeline (run once, or after changing the source docs):
```powershell
uv run src/s3_chunk_docs.py data/website/content/en/docs/concepts
uv run src/s4_embed_docs.py data/website/content/en/docs/concepts
```

Ask questions:
```powershell
uv run src/cli.py
```

## What's next (Phase 4+)

- Evaluation harness — a test set of 20-50 Q&A pairs with ground truth, scored automatically via
  RAGAS (faithfulness, answer relevance, context precision/recall)
- Systematic before/after comparison of semantic-only vs. hybrid vs. hybrid+reranked across the
  full test set, not just one example query
- Iteration on chunk size, top_k, and prompt wording based on measured eval results
- FastAPI service, caching, logging/observability, Docker packaging

## License

MIT