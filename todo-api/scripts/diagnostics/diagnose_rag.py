
from storage import read_excel
from rag.retriever import retrieve_tasks

# Todas las tareas guardadas en Excel
excel_tasks = read_excel()

print("\n--- TAREAS EN EXCEL ---")

for task in excel_tasks:
    print(f"ID: {task.id} | Nombre: {task.name}")
    print(f"Descripción: {task.description}")
    print(f"Estado: {task.status}\n")

# Tareas recuperadas mediante búsqueda semántica
print("\n--- RESULTADOS DEL RAG ---")

rag_tasks = retrieve_tasks(
    query="CRUD",
    limit=10,
)

for task in rag_tasks:
    print(f"ID: {task.id} | Nombre: {task.name}")
    print(f"Descripción: {task.description}")
    print(f"Estado: {task.status}\n")

print("--- RESUMEN ---")
print(f"Tareas en Excel: {len(excel_tasks)}")
print(f"Tareas recuperadas: {len(rag_tasks)}")
