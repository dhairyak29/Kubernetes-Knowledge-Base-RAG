"""
Step 4: Embed
Embeds all chunks using sentence-transformers and stores them in a
persistent Chroma vector database, along with metadata needed for
citations later.
 
Usage:
    python embed_docs.py /path/to/concepts
    python embed_docs.py /path/to/concepts --query "how does pod networking work"
"""

import sys
 
import chromadb
from sentence_transformers import SentenceTransformer
 
from s1_load_docs import load_markdown_docs
from s2_clean_docs import clean_all
from s3_chunk_docs import chunk_all, EMBEDDING_MODEL_NAME
 
# --- Config -----------------------------------------------------------
 
CHROMA_PERSIST_DIR = "./.chroma"
COLLECTION_NAME = "k8s-concepts"
BATCH_SIZE = 64


def build_vector_store(folder: str) -> chromadb.Collection:
    print("Loading and processing documents...")
    raw_docs = load_markdown_docs(folder)
    cleaned_docs = clean_all(raw_docs)
    chunks = chunk_all(cleaned_docs)
    print(f"  {len(chunks)} chunks ready to embed\n")
 
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
    print("Connecting to Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
 
    # Fresh start each run — for a portfolio project, re-embedding from
    # scratch is simpler and safer than trying to diff/update in place.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_title": c.doc_title,
                    "source_file": c.source_file,
                    "section_header": c.section_header,
                    "token_count": c.token_count,
                }
                for c in batch
            ],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
    return collection


def test_query(collection: chromadb.Collection, query: str, top_k: int = 5) -> None:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode([query]).tolist()
 
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
 
    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} results:\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        doc_preview = results["documents"][0][i][:150].replace("\n", " ")
        print(f"  [{i+1}] distance={distance:.4f}  {meta['doc_title']} > {meta['section_header']}")
        print(f"      {doc_preview}...\n")
 

 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python embed_docs.py /path/to/concepts")
        print('  python embed_docs.py /path/to/concepts --query "your question"')
        sys.exit(1)
 
    folder = sys.argv[1]
    coll = build_vector_store(folder)
 
    if "--query" in sys.argv:
        q = sys.argv[sys.argv.index("--query") + 1]
        test_query(coll, q)
    else:
        # default sanity-check query so you always see something at the end
        test_query(coll, "how does pod to pod communication work")


