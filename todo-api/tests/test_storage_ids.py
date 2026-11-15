import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from openpyxl import Workbook, load_workbook

import storage
from schemas import TaskSchema


class StorageIdTests(unittest.TestCase):
    def test_excel_operations_hold_the_shared_lock(self) -> None:
        task = TaskSchema(
            name="New task",
            description=None,
            date=date(2026, 11, 7),
            status="Done",
        )

        def load_locked_workbook(*args: object, **kwargs: object) -> MagicMock:
            self.assertTrue(storage._excel_lock.locked())
            workbook = MagicMock()
            workbook.__getitem__.return_value.iter_rows.return_value = []
            return workbook

        with patch.object(storage, "load_workbook", side_effect=load_locked_workbook):
            storage.read_excel()
            storage.get_tasks(10)
            storage.get_task(1)
            storage.create_task(task)
            storage.update_task(1, task)
            storage.delete_task(1)
            storage.patch_task(1, storage.TaskUpdate(name="Updated task"))

        with TemporaryDirectory() as directory:
            workbook_path = Path(directory) / "tasks.xlsx"
            with patch.object(storage, "EXCEL_FILE", workbook_path):
                workbook = MagicMock()

                def create_locked_workbook() -> MagicMock:
                    self.assertTrue(storage._excel_lock.locked())
                    return workbook

                with patch.object(storage, "Workbook", side_effect=create_locked_workbook):
                    storage.initialize_excel()

                workbook.close.assert_called_once()

    def test_concurrent_creates_use_distinct_ids_from_excel(self) -> None:
        with TemporaryDirectory() as directory:
            workbook_path = Path(directory) / "tasks.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Tasks"
            sheet.append(storage.HEADERS)
            sheet.append([1, "Duplicate one", None, date(2026, 11, 7), "Done"])
            sheet.append([1, "Duplicate two", None, date(2026, 11, 7), "Done"])
            workbook.save(workbook_path)
            workbook.close()

            task = TaskSchema(
                name="New task",
                description=None,
                date=date(2026, 11, 7),
                status="Done",
            )
            with patch.object(storage, "EXCEL_FILE", workbook_path):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    created = list(executor.map(lambda _: storage.create_task(task), range(2)))

                created_ids = [record.id for record in created]
                self.assertEqual(created_ids, [2, 3])
                self.assertEqual(len(created_ids), len(set(created_ids)))

                workbook = load_workbook(workbook_path, read_only=True, data_only=True)
                try:
                    sheet = workbook["Tasks"]
                    all_ids = [
                        row[0]
                        for row in sheet.iter_rows(
                            min_row=2,
                            min_col=1,
                            max_col=1,
                            values_only=True,
                        )
                    ]
                finally:
                    workbook.close()

            self.assertTrue(set(created_ids).isdisjoint({1}))
            self.assertCountEqual(all_ids, [1, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
