
from schemas import TaskRecord
from storage import get_task

from rag.embeddings import generate_embedding
from rag.vector_store import search_tasks


def retrieve_tasks(
    query: str,
    limit: int = 3,
    status: str | None = None,
) -> list[TaskRecord]:

    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía")

    if limit < 1:
        raise ValueError("El límite debe ser mayor que cero")

    # 1. Convertir la pregunta en un embedding
    query_embedding = generate_embedding(query)

    # 2. Buscar tareas similares en Qdrant
    results = search_tasks(
        query_embedding=query_embedding,
        limit=limit,
    )

    retrieved_tasks = []

    # 3. Recuperar los registros actuales desde Excel
    for result in results:

        if not isinstance(result.id, int):
            continue

        task = get_task(result.id)

        # 4. Ignorar tareas que ya no existen en Excel
        if task is None:
            continue

        # 5. Aplicar el filtro usando el estado actual de Excel
        if status is not None and task.status != status:
            continue

        retrieved_tasks.append(task)

    return retrieved_tasks
