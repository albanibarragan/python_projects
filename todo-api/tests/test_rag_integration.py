import hashlib
import re
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qdrant_client import QdrantClient

import storage
from rag import indexer, retriever, synchronizer, vector_store
from rag.indexer import task_to_document
from schemas import TaskSchema


def deterministic_embedding(text: str) -> list[float]:
    vector = [0.0] * 128
    for token in re.findall(r"\w+", text.casefold()):
        position = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4]) % len(vector)
        vector[position] += 1.0
    return vector


class RagLifecycleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_path = Path(self.temporary_directory.name)

        self.excel_file_patcher = patch.object(
            storage,
            "EXCEL_FILE",
            temporary_path / "tasks.xlsx",
        )
        self.excel_file_patcher.start()
        self.addCleanup(self.excel_file_patcher.stop)

        self.qdrant_client = QdrantClient(path=str(temporary_path / "qdrant"))
        self.addCleanup(self.qdrant_client.close)
        self.qdrant_client_patcher = patch.object(
            vector_store,
            "client",
            self.qdrant_client,
        )
        self.qdrant_client_patcher.start()
        self.addCleanup(self.qdrant_client_patcher.stop)

        self.index_embedding_patcher = patch.object(
            indexer,
            "generate_embedding",
            side_effect=deterministic_embedding,
        )
        self.index_embedding_patcher.start()
        self.addCleanup(self.index_embedding_patcher.stop)

        self.query_embedding_patcher = patch.object(
            retriever,
            "generate_embedding",
            side_effect=deterministic_embedding,
        )
        self.query_embedding_patcher.start()
        self.addCleanup(self.query_embedding_patcher.stop)

        storage.initialize_excel()

    def test_create_sync_retrieve_update_delete_lifecycle(self) -> None:
        tasks = [
            storage.create_task(
                TaskSchema(
                    name="Repasar Python",
                    description="Practicar funciones y tipos de Python.",
                    date=date(2026, 11, 15),
                    status="In progress",
                )
            ),
            storage.create_task(
                TaskSchema(
                    name="Crear API CRUD",
                    description="Construir una API REST con FastAPI.",
                    date=date(2026, 11, 15),
                    status="In progress",
                )
            ),
            storage.create_task(
                TaskSchema(
                    name="Explorar Qdrant",
                    description="Probar búsqueda semántica y embeddings.",
                    date=date(2026, 11, 15),
                    status="In progress",
                )
            ),
        ]
        task_ids = {task.id for task in tasks}

        synchronizer.reconcile_tasks()
        self.assertEqual(set(vector_store.list_indexed_tasks()), task_ids)

        initial_results = retriever.retrieve_tasks("API CRUD FastAPI", limit=3)
        self.assertEqual(initial_results[0].id, tasks[1].id)

        original_point = self.qdrant_client.retrieve(
            collection_name=vector_store.COLLECTION_NAME,
            ids=[tasks[1].id],
            with_payload=True,
            with_vectors=True,
        )[0]
        original_vector = original_point.vector

        updated_task = storage.update_task(
            tasks[1].id,
            TaskSchema(
                name="Ajustar HNSW ef_construct",
                description="Optimizar índice ANN para consultas vectoriales.",
                date=date(2026, 11, 15),
                status="Done",
            ),
        )
        self.assertIsNotNone(updated_task)

        synchronizer.reconcile_tasks()
        updated_point = self.qdrant_client.retrieve(
            collection_name=vector_store.COLLECTION_NAME,
            ids=[tasks[1].id],
            with_payload=True,
            with_vectors=True,
        )[0]
        self.assertEqual(
            updated_point.payload["document"],
            task_to_document(updated_task),
        )
        self.assertEqual(updated_point.payload["status"], "Done")
        self.assertNotEqual(updated_point.vector, original_vector)

        updated_results = retriever.retrieve_tasks("HNSW ef_construct", limit=3)
        self.assertEqual(updated_results[0].id, tasks[1].id)

        deleted_task = storage.delete_task(tasks[2].id)
        self.assertIsNotNone(deleted_task)
        synchronizer.reconcile_tasks()

        indexed_ids = set(vector_store.list_indexed_tasks())
        self.assertEqual(indexed_ids, task_ids - {tasks[2].id})
        remaining_results = retriever.retrieve_tasks(
            "Qdrant búsqueda semántica",
            limit=3,
        )
        self.assertNotIn(tasks[2].id, {task.id for task in remaining_results})


if __name__ == "__main__":
    unittest.main()
