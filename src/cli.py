"""
Minimal interactive CLI for the RAG pipeline — Phase 2's testing interface.
 
Loads everything once, then lets you ask questions repeatedly without
the per-question startup cost of reloading the embedding model each time.
 
Usage:
    python cli.py
"""
 
from s6_generate import generate_answer, print_result
 
 
BANNER = """
Kubernetes Docs RAG — ask a question (type 'exit' or 'quit' to stop)
----------------------------------------------------------------------
"""


def main():
    print(BANNER)
    while True:
        try:
            query = input("\nYour Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting")
            break
        result = generate_answer(query)
        print_result(result)

if __name__ == "__main__":
    main()