
from schemas import TaskRecord
from rag.embeddings import generate_embedding
from rag.vector_store import save_task, update_task_vector, delete_task


def task_to_document(task: TaskRecord) -> str:
    return (
        f"Tarea: {task.name}\n"
        f"Descripción: {task.description or 'Sin descripción'}\n"
        f"Fecha: {task.date.isoformat()}\n"
        f"Estado: {task.status}"
    )


def index_task(task: TaskRecord, update: bool = False) -> None:
    document = task_to_document(task)
    embedding = generate_embedding(document)

    if update:
        update_task_vector(
            task_id=task.id,
            document=document,
            embedding=embedding,
            status=task.status,
        )
    else:
        save_task(
            task_id=task.id,
            document=document,
            embedding=embedding,
            status=task.status,
        )


def remove_task_from_index(task_id: int) -> None:
    delete_task(task_id)
