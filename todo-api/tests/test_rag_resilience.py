import hashlib
import re
import unittest
import asyncio
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import main
import storage
from rag import indexer, retriever, synchronizer, vector_store
from schemas import TaskSchema


def deterministic_embedding(text: str) -> list[float]:
    vector = [0.0] * 128
    for token in re.findall(r"\w+", text.casefold()):
        position = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4]) % len(vector)
        vector[position] += 1.0
    return vector


class RagResilienceTests(unittest.TestCase):
    def test_qdrant_failure_keeps_excel_and_retries_on_next_startup(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            excel_patcher = patch.object(
                storage,
                "EXCEL_FILE",
                temporary_path / "tasks.xlsx",
            )
            qdrant_client = QdrantClient(path=str(temporary_path / "qdrant"))
            qdrant_patcher = patch.object(vector_store, "client", qdrant_client)
            index_embedding_patcher = patch.object(
                indexer,
                "generate_embedding",
                side_effect=deterministic_embedding,
            )
            query_embedding_patcher = patch.object(
                retriever,
                "generate_embedding",
                side_effect=deterministic_embedding,
            )

            excel_patcher.start()
            qdrant_patcher.start()
            index_embedding_patcher.start()
            query_embedding_patcher.start()
            try:
                storage.initialize_excel()
                task = storage.create_task(
                    TaskSchema(
                        name="Verify Qdrant retry",
                        description="Check recovery after temporary Qdrant failure.",
                        date=date(2026, 11, 15),
                        status="In progress",
                    )
                )

                with (
                    patch.object(
                        qdrant_client,
                        "upsert",
                        side_effect=RuntimeError("Qdrant unavailable"),
                    ),
                    self.assertLogs(synchronizer.logger, level="ERROR"),
                ):
                    with TestClient(main.app) as client:
                        self.assertEqual(
                            client.get("/health").json(),
                            {"status": "healthy"},
                        )

                self.assertEqual(storage.get_task(task.id), task)
                self.assertEqual(vector_store.list_indexed_tasks(), {})

                with TestClient(main.app) as client:
                    self.assertEqual(
                        client.get("/health").json(),
                        {"status": "healthy"},
                    )

                self.assertEqual(set(vector_store.list_indexed_tasks()), {task.id})
                retrieved_tasks = retriever.retrieve_tasks(
                    "Qdrant recovery retry",
                    limit=1,
                )
                self.assertEqual([item.id for item in retrieved_tasks], [task.id])
            finally:
                query_embedding_patcher.stop()
                index_embedding_patcher.stop()
                qdrant_patcher.stop()
                excel_patcher.stop()
                qdrant_client.close()


class PeriodicReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def test_periodic_reconciliation_retries_after_a_failure(self) -> None:
        with (
            patch.object(main.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            patch.object(main, "run_in_threadpool", new_callable=AsyncMock) as run_sync,
            patch.object(main, "reconcile_tasks") as reconcile,
            self.assertLogs(main.logger, level="ERROR"),
        ):
            sleep.side_effect = [None, None, asyncio.CancelledError]
            run_sync.side_effect = [RuntimeError("Qdrant unavailable"), None]

            with self.assertRaises(asyncio.CancelledError):
                await main._reconcile_periodically()

        self.assertEqual(run_sync.await_count, 2)
        self.assertEqual(run_sync.await_args.args, (reconcile,))
        self.assertEqual(
            sleep.await_args_list[0].args,
            (main.RECONCILIATION_INTERVAL_SECONDS,),
        )
        self.assertEqual(
            main.RECONCILIATION_INTERVAL_SECONDS,
            5 * 60,
        )


if __name__ == "__main__":
    unittest.main()
