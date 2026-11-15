from fastapi import APIRouter, HTTPException, Query
import logging

import storage
from schemas import (
    TaskListResponse,
    TaskMutationResponse,
    TaskRecord,
    TaskSchema,
    TaskSummary,
    TaskUpdate,
)

from rag.indexer import index_task, remove_task_from_index

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
)


@router.get("")
def get_tasks(
    limit: int = Query(default=10, ge=1, le=100),
) -> TaskListResponse:
    return TaskListResponse(tasks=storage.get_tasks(limit), limit=limit)


@router.get("/{task_id}")
def get_task(
    task_id: int, include_desc: bool = False
) -> TaskRecord | TaskSummary:
    task = storage.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if include_desc:
        return task
    return TaskSummary.model_validate(task.model_dump())


@router.post("", status_code=201, response_model=TaskMutationResponse)
def create_task(task: TaskSchema) -> TaskMutationResponse:
    created_task = storage.create_task(task)

    try:
        index_task(created_task)
    except Exception:
        logger.exception(
            "No se pudo indexar la tarea %s",
            created_task.id,
        )

    return TaskMutationResponse(
        message="Task created successfully",
        task=created_task,
    )


@router.delete("/{task_id}", response_model=TaskMutationResponse)
def delete_task(task_id: int) -> TaskMutationResponse:
    deleted_task = storage.delete_task(task_id)

    if deleted_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        remove_task_from_index(task_id)
    except Exception:
        logger.exception(
            "No se pudo eliminar el vector de la tarea %s",
            task_id,
        )

    return TaskMutationResponse(
        message="Task deleted successfully",
        task=deleted_task,
    )

@router.put("/{task_id}", response_model=TaskRecord)
def update_task(task_id: int, task_data: TaskSchema) -> TaskRecord:
    updated_task = storage.update_task(task_id, task_data)

    if updated_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        index_task(updated_task, update=True)
    except Exception:
        logger.exception(
            "No se pudo actualizar el vector de la tarea %s",
            task_id,
        )

    return updated_task


@router.patch("/{task_id}", response_model=TaskRecord)
def patch_task(task_id: int, task_data: TaskUpdate) -> TaskRecord:
    updated_task = storage.patch_task(task_id, task_data)

    if updated_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        index_task(updated_task, update=True)
    except Exception:
        logger.exception(
            "No se pudo actualizar el vector de la tarea %s",
            task_id,
        )

    return updated_task