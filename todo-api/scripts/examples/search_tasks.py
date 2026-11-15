from rag.embeddings import generate_embedding
from rag.vector_store import search_tasks

query = "Quiero estudiar programación"

query_embedding = generate_embedding(query)

results = search_tasks(
    query_embedding=query_embedding,
    limit=5,
    status="Complete",
)

for result in results:
    print("ID:", result.id)
    print("Similitud:", result.score)
    print("Estado:", result.payload["status"])
    print("Documento:", result.payload["document"])
    print("-------------------")