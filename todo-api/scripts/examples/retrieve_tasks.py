
from rag.retriever import retrieve_tasks

query = "Quiero repasar cómo crear una API con Python"

tasks = retrieve_tasks(
    query=query,
    limit=3,
)

if not tasks:
    print("No se encontraron tareas relacionadas.")

for task in tasks:
    print("ID:", task.id)
    print("Nombre:", task.name)
    print("Descripción:", task.description)
    print("Fecha:", task.date)
    print("Estado:", task.status)
    print("-------------------")
