def main():
    print("Hello from rag-kubernetes-docs!")


from fastapi import FastAPI
from src.s6_generate import generate_answer

app = FastAPI()

@app.get("/get")
async def get_kubernetes_data(query: str):
    return generate_answer(query)


if __name__ == "__main__":
    main()
