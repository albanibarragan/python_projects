from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

import storage
from rag.indexer import index_task
from rag.retriever import retrieve_tasks
from rag import vector_store
from schemas import TaskSchema


TEST_TASKS = (
    TaskSchema(
        name="Repasar Python avanzado",
        description="Estudiar funciones, tipos, módulos y buenas prácticas de Python.",
        date=date(2026, 11, 15),
        status="In progress",
    ),
    TaskSchema(
        name="Practicar CRUD con FastAPI",
        description=(
            "Crear, consultar, actualizar y eliminar registros en una API REST "
            "con FastAPI."
        ),
        date=date(2026, 11, 15),
        status="In progress",
    ),
    TaskSchema(
        name="Estudiar bases de datos vectoriales",
        description="Aprender embeddings, búsqueda semántica y Qdrant.",
        date=date(2026, 11, 15),
        status="In progress",
    ),
)

TEST_CASES = (
    ("¿Qué tarea tengo relacionada con CRUD?", 2),
    ("¿Qué debo repasar sobre Python?", 1),
    ("¿Qué tareas tengo sobre bases de datos vectoriales?", 3),
)


def main() -> int:
    original_excel_file = storage.EXCEL_FILE
    original_qdrant_client = vector_store.client

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        test_client = QdrantClient(path=str(temporary_path / "qdrant"))

        try:
            storage.EXCEL_FILE = temporary_path / "tasks.xlsx"
            vector_store.client = test_client
            storage.initialize_excel()

            tasks = [storage.create_task(task) for task in TEST_TASKS]
            for task in tasks:
                index_task(task)

            print(
                "Prueba aislada: usa tareas y Qdrant temporales; "
                "no modifica tus datos reales."
            )

            failures = 0
            for question, expected_id in TEST_CASES:
                retrieved_tasks = retrieve_tasks(query=question, limit=len(tasks))
                retrieved_ids = [task.id for task in retrieved_tasks]
                top_id = retrieved_ids[0] if retrieved_ids else None
                passed = top_id == expected_id
                failures += not passed

                print(f"\nPregunta: {question}")
                for task in retrieved_tasks:
                    print(f"  ID: {task.id} | {task.name}")
                print(f"ID esperado en primera posición: {expected_id}")
                print(
                    f"Resultado: {'CORRECTO' if passed else 'REVISAR'} "
                    f"(primer resultado: {top_id})"
                )

            print(f"\nResumen: {len(TEST_CASES) - failures}/{len(TEST_CASES)} correctas")
            return int(failures > 0)
        finally:
            vector_store.client = original_qdrant_client
            storage.EXCEL_FILE = original_excel_file
            test_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
