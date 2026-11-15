
from storage import read_excel
from rag.indexer import index_task

tasks = read_excel()

print(f"Se encontraron {len(tasks)} tareas en Excel.\n")

for task in tasks:
    try:
        index_task(task)
        print(f"Indexada correctamente: ID {task.id} - {task.name}")

    except Exception as exc:
        print(f"Error al indexar ID {task.id}: {exc}")

print("\nProceso de indexación finalizado.")
