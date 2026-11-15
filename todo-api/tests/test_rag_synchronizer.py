import unittest
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from rag import synchronizer
from rag.indexer import task_to_document
from schemas import TaskRecord


class ReconcileTasksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            TaskRecord(
                id=1,
                name="Unchanged",
                description="Already indexed",
                date=date(2026, 11, 14),
                status="In progress",
            ),
            TaskRecord(
                id=2,
                name="Updated",
                description="Changed in Excel",
                date=date(2026, 11, 14),
                status="Done",
            ),
            TaskRecord(
                id=3,
                name="Missing",
                description="Not yet indexed",
                date=date(2026, 11, 14),
                status="Complete",
            ),
        ]

    def test_indexes_missing_and_changed_tasks_and_removes_orphans(self) -> None:
        indexed_tasks = {
            1: {
                "document": task_to_document(self.tasks[0]),
                "status": self.tasks[0].status,
            },
            2: {
                "document": "Old task document",
                "status": "In progress",
            },
            4: {
                "document": "Deleted task",
                "status": "Done",
            },
        }

        with (
            patch.object(synchronizer.storage, "read_excel", return_value=self.tasks),
            patch.object(
                synchronizer,
                "list_indexed_tasks",
                return_value=indexed_tasks,
            ),
            patch.object(synchronizer, "index_task") as index_task,
            patch.object(synchronizer, "remove_task_from_index") as remove_task,
        ):
            synchronizer.reconcile_tasks()

        self.assertEqual(
            index_task.call_args_list,
            [
                unittest.mock.call(self.tasks[1], update=True),
                unittest.mock.call(self.tasks[2], update=False),
            ],
        )
        remove_task.assert_called_once_with(4)

    def test_continues_reconciling_after_an_indexing_failure(self) -> None:
        with (
            patch.object(synchronizer.storage, "read_excel", return_value=self.tasks),
            patch.object(synchronizer, "list_indexed_tasks", return_value={}),
            patch.object(
                synchronizer,
                "index_task",
                side_effect=[RuntimeError("embedding unavailable"), None, None],
            ) as index_task,
            patch.object(synchronizer, "remove_task_from_index"),
            self.assertLogs(synchronizer.logger, level="ERROR"),
        ):
            synchronizer.reconcile_tasks()

        self.assertEqual(index_task.call_count, len(self.tasks))

    def test_startup_reconciliation_failure_does_not_disable_api(self) -> None:
        with (
            patch.object(
                main,
                "reconcile_tasks",
                side_effect=RuntimeError("Qdrant unavailable"),
            ),
            TestClient(main.app) as client,
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})


if __name__ == "__main__":
    unittest.main()
