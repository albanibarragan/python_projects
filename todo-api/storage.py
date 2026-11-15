from datetime import date
from threading import Lock

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from schemas import TaskRecord, TaskSchema, TaskUpdate
from pathlib import Path

EXCEL_FILE = Path(__file__).resolve().parent / "tasks.xlsx"
HEADERS = ["id", "name", "description", "date", "status"]
_excel_lock = Lock()

def initialize_excel():
    with _excel_lock:
        if not EXCEL_FILE.exists():
            workbook = Workbook()
            try:
                sheet = workbook.active
                sheet.title = "Tasks"

                sheet.append(HEADERS)

                workbook.save(EXCEL_FILE)
            finally:
                workbook.close()


initialize_excel()

def read_excel() -> list[TaskRecord]:
    with _excel_lock:
        workbook = load_workbook(EXCEL_FILE)
        try:
            sheet = workbook["Tasks"]

            tasks = []

            for row in sheet.iter_rows(min_row=2, values_only=True):
                task = TaskRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    date=row[3].date() if hasattr(row[3], "date") else row[3],
                    status=row[4],
                )

                tasks.append(task)

            return tasks
        finally:
            workbook.close()

_tasks: list[TaskRecord] = [
    TaskRecord(
        id=1,
        name="Inciializar mi proyecto en FastAPI",
        description="Estoy aprendiendo backend con FastAPI",
        date=date(2026, 11, 7),
        status="Complete",
    )
]
_next_id = max((task.id for task in _tasks), default=0) + 1


def get_tasks(limit: int) -> list[TaskRecord]:
    return read_excel()[:limit]


def get_task(task_id: int) -> TaskRecord | None:
    tasks = read_excel()
    return next((task for task in tasks if task.id == task_id), None)


def _read_existing_ids(sheet: Worksheet) -> list[int]:
    ids: list[int] = []
    for (value,) in sheet.iter_rows(
        min_row=2,
        min_col=1,
        max_col=1,
        values_only=True,
    ):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            ids.append(value)
        elif isinstance(value, float) and value.is_integer() and value > 0:
            ids.append(int(value))
    return ids


def create_task(task: TaskSchema) -> TaskRecord:
    with _excel_lock:
        workbook = load_workbook(EXCEL_FILE)
        try:
            sheet = workbook["Tasks"]
            next_id = max(_read_existing_ids(sheet), default=0) + 1

            sheet.append(
                [
                    next_id,
                    task.name,
                    task.description,
                    task.date,
                    task.status,
                ]
            )
            workbook.save(EXCEL_FILE)
        finally:
            workbook.close()

    return TaskRecord(id=next_id, **task.model_dump())


def update_task(task_id: int, task_data: TaskSchema) -> TaskRecord | None:
    with _excel_lock:
        workbook = load_workbook(EXCEL_FILE)
        try:
            sheet = workbook["Tasks"]

            for row in sheet.iter_rows(min_row=2):
                if row[0].value == task_id:

                    row[1].value = task_data.name
                    row[2].value = task_data.description
                    row[3].value = task_data.date
                    row[4].value = task_data.status

                    updated_task = TaskRecord(
                        id=task_id,
                        **task_data.model_dump()
                    )

                    workbook.save(EXCEL_FILE)

                    return updated_task

            return None
        finally:
            workbook.close()


def delete_task(task_id: int) -> TaskRecord | None:
    with _excel_lock:
        workbook = load_workbook(EXCEL_FILE)
        try:
            sheet = workbook["Tasks"]

            for row in sheet.iter_rows(min_row=2):
                if row[0].value == task_id:
                    task_date = row[3].value
                    if hasattr(task_date, "date"):
                        task_date = task_date.date()

                    deleted_task = TaskRecord(
                        id=row[0].value,
                        name=row[1].value,
                        description=row[2].value,
                        date=task_date,
                        status=row[4].value,
                    )

                    sheet.delete_rows(row[0].row)
                    workbook.save(EXCEL_FILE)
                    return deleted_task

            return None
        finally:
            workbook.close()


def patch_task(task_id: int, task_data: TaskUpdate) -> TaskRecord | None:
    with _excel_lock:
        workbook = load_workbook(EXCEL_FILE)
        try:
            sheet = workbook["Tasks"]
            updates = task_data.model_dump(exclude_unset=True)
            fields = {
                "name": 1,
                "description": 2,
                "date": 3,
                "status": 4,
            }

            for row in sheet.iter_rows(min_row=2):
                if row[0].value != task_id:
                    continue

                for field, value in updates.items():
                    row[fields[field]].value = value

                date_value = row[3].value
                updated_task = TaskRecord(
                    id=row[0].value,
                    name=row[1].value,
                    description=row[2].value,
                    date=date_value.date() if hasattr(date_value, "date") else date_value,
                    status=row[4].value,
                )
                workbook.save(EXCEL_FILE)
                return updated_task

            return None
        finally:
            workbook.close()
