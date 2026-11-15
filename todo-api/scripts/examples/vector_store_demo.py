from rag.embeddings import generate_embedding
from rag.vector_store import save_task, search_tasks


tasks = [
    {
        "id": 1,
        "name": "Aprender Python",
        "description": "Estudiar programación con Python",
        "status": "In progress",
    },
    {
        "id": 2,
        "name": "Crear API REST",
        "description": "Desarrollar una API con FastAPI",
        "status": "In progress",
    },
    {
        "id": 3,
        "name": "Comprar comida",
        "description": "Ir al supermercado a comprar alimentos",
        "status": "Complete",
    },
]

# Guardar tareas en Qdrant
for task in tasks:
    document = (
        f"Tarea: {task['name']}. "
        f"Descripción: {task['description']}. "
        f"Estado: {task['status']}."
    )

    embedding = generate_embedding(document)

    save_task(
        task_id=task["id"],
        document=document,
        embedding=embedding,
        status=task["status"],
    )

# Realizar búsqueda semántica
question = "¿Qué tareas tengo relacionadas con programación?"

query_embedding = generate_embedding(question)

results = search_tasks(query_embedding, limit=3)

# Mostrar resultados
for result in results:
    payload = result.payload or {}
    print(f"Similitud: {result.score:.4f}")
    print(f"Tarea: {payload['document']}")
    print("-" * 40)