"""
Generation module — Phase 2's final piece: takes a user question, retrieves
relevant chunks, and calls Groq to generate an answer grounded in those
chunks only.
 
Requires GROQ_API_KEY in your .env file.
 
Usage (standalone test):
    python generate.py "how does pod to pod communication work"
"""
 
import os
import sys
 
from dotenv import load_dotenv
from groq import Groq

from src.s5_retrieve import retrieve, format_for_prompt
# from hybrid_search import hybrid_search
from src.reranker import retrieve_and_rerank

load_dotenv()
 
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
 
SYSTEM_PROMPT = """You are a documentation assistant for Kubernetes. \
Answer the user's question using ONLY the information in the provided sources below.
 
Rules:
- Do not use any knowledge you have outside of the provided sources, even if you know the answer.
- If the sources do not contain enough information to answer the question, say so clearly \
instead of guessing.
- When you state a fact, refer to which source it came from, e.g. "(Source 2)".
- Be concise and technically precise. Prefer direct answers over padding.
"""

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(query: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG generation: retrieve -> build prompt -> call Groq -> return
    the answer along with the chunks used, so callers can display citations.
    """

    chunks = retrieve_and_rerank(query=query, top_k=top_k)

    if not chunks:
        return{
            "answer": "No relevant information was found in the knowledge base.",
            "chunks": []
        }

    context = format_for_prompt(chunks=chunks)

    user_message = f"""Sources:{context}
    Question = {query}"""

    client = _get_client()

    response = client.chat.completions.create(
        model= GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content" : user_message}
        ],
    temperature= 0.1
    )

    answer = response.choices[0].message.content

    return{
        "answer": answer,
        "chunks": chunks,
    }


def print_result(result: dict) -> None:
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources used:")
    for i, c in enumerate(result["chunks"], 1):
        print(f"  [{i}] {c.doc_title} > {c.section_header} > (distance= {c.distance:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python s6_generate.py "your query"')
        sys.exit(1)

    query = sys.argv[1]
    result = generate_answer(query=query)
    print_result(result)

    

