import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
import main
import storage
import routers.tasks as tasks_router
from rag.embeddings import EmbeddingError


class TaskApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.excel_file_patcher = patch.object(
            storage,
            "EXCEL_FILE",
            Path(self.temporary_directory.name) / "tasks.xlsx",
        )
        self.excel_file_patcher.start()
        storage.initialize_excel()

        self.startup_sync_patcher = patch.object(main, "reconcile_tasks")
        self.startup_sync_patcher.start()
        self.index_task_patcher = patch.object(tasks_router, "index_task")
        self.index_task_patcher.start()
        self.remove_task_patcher = patch.object(
            tasks_router,
            "remove_task_from_index",
        )
        self.remove_task_patcher.start()

        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.remove_task_patcher.stop()
        self.index_task_patcher.stop()
        self.startup_sync_patcher.stop()
        self.excel_file_patcher.stop()
        self.temporary_directory.cleanup()

    def create_task(self, name: str = "New task") -> dict[str, Any]:
        response = self.client.post(
            "/tasks",
            json={
                "name": name,
                "description": "A task description",
                "date": "2026-11-07",
                "status": "In progress",
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.json()["task"]

    def test_create_task_returns_validated_record(self) -> None:
        task = self.create_task()
        self.assertGreater(task["id"], 0)
        self.assertEqual(task["name"], "New task")
        self.client.delete(f"/tasks/{task['id']}")

    def test_create_rejects_invalid_input_and_extra_fields(self) -> None:
        payload: dict[str, Any] = {
            "name": "New task",
            "description": None,
            "date": "2026-11-07",
            "status": "Pending",
        }
        self.assertEqual(self.client.post("/tasks", json=payload).status_code, 422)

        payload["status"] = "Done"
        payload["id"] = 99
        self.assertEqual(self.client.post("/tasks", json=payload).status_code, 422)

    def test_list_limit_is_bounded(self) -> None:
        self.assertEqual(self.client.get("/tasks?limit=100").status_code, 200)
        self.assertEqual(self.client.get("/tasks?limit=101").status_code, 422)
        self.assertEqual(self.client.get("/tasks?limit=0").status_code, 422)

    def test_get_and_delete_tasks(self) -> None:
        created_task = self.create_task()
        task_id = created_task["id"]
        response = self.client.get(f"/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("description", response.json())

        full_response = self.client.get(f"/tasks/{task_id}?include_desc=true")
        self.assertEqual(full_response.json()["description"], "A task description")

        self.assertEqual(self.client.delete(f"/tasks/{task_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/tasks/{task_id}").status_code, 404)
        self.assertEqual(self.client.delete("/tasks/999").status_code, 404)

    def test_put_replaces_task_and_preserves_id(self) -> None:
        created_task = self.create_task()
        task_id = created_task["id"]
        response = self.client.put(
            f"/tasks/{task_id}",
            json={
                "name": "Updated task",
                "description": "Updated description",
                "date": "2026-11-08",
                "status": "Done",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "id": task_id,
                "name": "Updated task",
                "description": "Updated description",
                "date": "2026-11-08",
                "status": "Done",
            },
        )
        self.assertEqual(self.client.get(f"/tasks/{task_id}?include_desc=true").json(),
                         response.json())
        self.client.delete(f"/tasks/{task_id}")

    def test_put_keeps_excel_update_when_openrouter_embedding_fails(self) -> None:
        created_task = self.create_task()
        task_id = created_task["id"]

        with (
            patch.object(
                tasks_router,
                "index_task",
                side_effect=EmbeddingError("OpenRouter no responde"),
            ),
            self.assertLogs("routers.tasks", level="ERROR"),
        ):
            response = self.client.put(
                f"/tasks/{task_id}",
                json={
                    "name": "Updated despite embedding failure",
                    "description": "Excel remains the source of truth",
                    "date": "2026-11-08",
                    "status": "Done",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            storage.get_task(task_id).name,
            "Updated despite embedding failure",
        )

    def test_put_returns_404_for_missing_task(self) -> None:
        response = self.client.put(
            "/tasks/999",
            json={
                "name": "Updated task",
                "description": None,
                "date": "2026-11-08",
                "status": "Done",
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_patch_updates_only_provided_fields(self) -> None:
        created_task = self.create_task()
        task_id = created_task["id"]

        response = self.client.patch(
            f"/tasks/{task_id}",
            json={"name": "Renamed task", "description": None},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], task_id)
        self.assertEqual(response.json()["name"], "Renamed task")
        self.assertIsNone(response.json()["description"])
        self.assertEqual(response.json()["date"], created_task["date"])
        self.assertEqual(response.json()["status"], created_task["status"])
        self.client.delete(f"/tasks/{task_id}")

    def test_patch_rejects_null_required_fields_and_missing_tasks(self) -> None:
        created_task = self.create_task()
        task_id = created_task["id"]
        response = self.client.patch(f"/tasks/{task_id}", json={"name": None})
        self.assertEqual(response.status_code, 422)

        response = self.client.patch("/tasks/999", json={"name": "Renamed task"})
        self.assertEqual(response.status_code, 404)
        self.client.delete(f"/tasks/{task_id}")

    def test_create_task_when_storage_is_empty(self) -> None:
        existing_tasks = self.client.get("/tasks?limit=100").json()["tasks"]
        for task in existing_tasks:
            self.client.delete(f"/tasks/{task['id']}")

        created_task = self.create_task("First task")
        self.assertEqual(
            self.client.get("/tasks?limit=100").json()["tasks"],
            [created_task],
        )
        self.client.delete(f"/tasks/{created_task['id']}")


if __name__ == "__main__":
    unittest.main()
