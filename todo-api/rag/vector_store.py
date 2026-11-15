import atexit
import os
from pathlib import Path
from threading import Lock
from uuid import UUID

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = Path(
    os.getenv("QDRANT_PATH")
    or str(Path(__file__).resolve().parents[1] / "qdrant_db")
)
COLLECTION_NAME = "tasks"

if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(path=str(QDRANT_PATH))
atexit.register(client.close)
_collection_lock = Lock()


def _ensure_collection(vector_size: int) -> None:
    with _collection_lock:
        if client.collection_exists(COLLECTION_NAME):
            _check_collection_dimension(vector_size)
            return

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def _check_collection_dimension(vector_size: int) -> None:
    collection = client.get_collection(COLLECTION_NAME)
    configured_vectors = collection.config.params.vectors
    if not isinstance(configured_vectors, models.VectorParams):
        raise ValueError(
            f"La colección '{COLLECTION_NAME}' debe usar un único vector sin nombre"
        )
    if configured_vectors.size != vector_size:
        raise ValueError(
            f"La colección '{COLLECTION_NAME}' espera vectores de "
            f"{configured_vectors.size} dimensiones, pero se recibieron {vector_size}"
        )


def _upsert_task(
    task_id: int,
    document: str,
    embedding: list[float],
    status: str,
) -> None:
    if not embedding:
        raise ValueError("embedding must contain at least one value")

    _ensure_collection(len(embedding))
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=task_id,
                vector=embedding,
                payload={"document": document, "status": status},
            )
        ],
    )


def save_task(
    task_id: int,
    document: str,
    embedding: list[float],
    status: str,
) -> None:
    _upsert_task(task_id, document, embedding, status)


def update_task_vector(
    task_id: int,
    document: str,
    embedding: list[float],
    status: str,
) -> None:
    _upsert_task(task_id, document, embedding, status)


def search_tasks(
    query_embedding: list[float],
    limit: int = 3,
    status: str | None = None,
) -> list[models.ScoredPoint]:
    if not query_embedding:
        raise ValueError("query_embedding must contain at least one value")
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if not client.collection_exists(COLLECTION_NAME):
        return []

    _check_collection_dimension(len(query_embedding))
    query_filter = None
    if status is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(value=status),
                )
            ]
        )

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    ).points


def list_indexed_tasks() -> dict[int, dict[str, object]]:
    if not client.collection_exists(COLLECTION_NAME):
        return {}

    indexed_tasks: dict[int, dict[str, object]] = {}
    offset: int | str | UUID | None = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            if not isinstance(point.id, int):
                raise ValueError(
                    f"El ID del punto de Qdrant no es un entero: {point.id}"
                )
            indexed_tasks[point.id] = point.payload or {}

        if offset is None:
            return indexed_tasks


def delete_task(task_id: int) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=[task_id]),
    )
