import logging

import storage
from rag.indexer import index_task, remove_task_from_index, task_to_document
from rag.vector_store import list_indexed_tasks


logger = logging.getLogger(__name__)


def reconcile_tasks() -> None:
    tasks = storage.read_excel()
    indexed_tasks = list_indexed_tasks()
    tasks_by_id = {task.id: task for task in tasks}
    indexed_count = 0
    removed_count = 0
    failed_count = 0

    for task in tasks:
        indexed_payload = indexed_tasks.get(task.id)
        document = task_to_document(task)
        if (
            indexed_payload is not None
            and indexed_payload.get("document") == document
            and indexed_payload.get("status") == task.status
        ):
            continue

        try:
            index_task(task, update=indexed_payload is not None)
            indexed_count += 1
        except Exception:
            failed_count += 1
            logger.exception(
                "No se pudo sincronizar la tarea %s con Qdrant",
                task.id,
            )

    for task_id in indexed_tasks:
        if task_id in tasks_by_id:
            continue

        try:
            remove_task_from_index(task_id)
            removed_count += 1
        except Exception:
            failed_count += 1
            logger.exception(
                "No se pudo eliminar de Qdrant la tarea inexistente %s",
                task_id,
            )

    logger.info(
        "Reconciliación Excel-Qdrant finalizada: %s indexadas/actualizadas, "
        "%s eliminadas, %s fallidas",
        indexed_count,
        removed_count,
        failed_count,
    )
