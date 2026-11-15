import unittest
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from rag import vector_store


class QdrantTaskStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_client = vector_store.client
        self.temporary_directory = TemporaryDirectory()
        vector_store.client = QdrantClient(path=self.temporary_directory.name)

    def tearDown(self) -> None:
        vector_store.client.close()
        vector_store.client = self.original_client
        self.temporary_directory.cleanup()

    def test_search_is_empty_before_first_save(self) -> None:
        self.assertEqual(vector_store.search_tasks([0.1, 0.2]), [])
        self.assertEqual(vector_store.list_indexed_tasks(), {})

    def test_upsert_replaces_existing_task(self) -> None:
        vector_store.save_task(1, "Original task", [0.1, 0.2], "In progress")
        vector_store.update_task_vector(1, "Updated task", [0.1, 0.2], "Done")

        results = vector_store.search_tasks([0.1, 0.2], limit=1)

        self.assertEqual(len(results), 1)
        payload = results[0].payload or {}
        self.assertEqual(payload["document"], "Updated task")
        self.assertEqual(payload["status"], "Done")

    def test_search_filters_by_status(self) -> None:
        vector_store.save_task(1, "Pending task", [1.0, 0.0], "In progress")
        vector_store.save_task(2, "Completed task", [0.9, 0.1], "Complete")

        results = vector_store.search_tasks([1.0, 0.0], status="In progress")

        self.assertEqual([result.id for result in results], [1])

    def test_rejects_vectors_with_wrong_collection_dimension(self) -> None:
        vector_store.save_task(1, "Task", [0.1, 0.2], "In progress")

        with self.assertRaisesRegex(ValueError, "espera vectores de 2 dimensiones"):
            vector_store.update_task_vector(
                1,
                "Updated task",
                [0.1, 0.2, 0.3],
                "Done",
            )

        with self.assertRaisesRegex(ValueError, "espera vectores de 2 dimensiones"):
            vector_store.search_tasks([0.1, 0.2, 0.3])

    def test_save_search_and_delete_task(self) -> None:
        vector_store.save_task(
            task_id=1,
            document="Aprender programación con Python",
            embedding=[0.1, 0.2, 0.3],
            status="In progress",
        )

        results = vector_store.search_tasks([0.1, 0.2, 0.3], limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, 1)
        self.assertEqual(
            results[0].payload,
            {
                "document": "Aprender programación con Python",
                "status": "In progress",
            },
        )
        self.assertEqual(
            vector_store.list_indexed_tasks(),
            {
                1: {
                    "document": "Aprender programación con Python",
                    "status": "In progress",
                }
            },
        )

        vector_store.delete_task(1)
        self.assertEqual(vector_store.search_tasks([0.1, 0.2, 0.3]), [])
        self.assertEqual(vector_store.list_indexed_tasks(), {})


if __name__ == "__main__":
    unittest.main()
