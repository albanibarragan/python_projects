
from rag.embeddings import generate_embedding
from rag.vector_store import search_tasks
from rag.retriever import retrieve_tasks
from rag.synchronizer import reconcile_tasks
from storage import read_excel

if not read_excel():
    print(
        "Excel no contiene tareas. Agrega o restaura las tareas antes de "
        "probar la búsqueda semántica."
    )
    raise SystemExit(0)

print("Sincronizando las tareas de Excel con Qdrant...")
reconcile_tasks()

questions = [
    "CRUD",
    "¿Qué tarea tengo relacionada con CRUD?",
    "Python",
    "¿Qué debo repasar sobre Python?",
    "bases de datos vectoriales",
    "¿Qué tareas tengo sobre bases de datos vectoriales?",
]

for question in questions:
    print("\n" + "=" * 60)
    print(f"PREGUNTA: {question}")

    try:
        # Primera etapa: generar embedding
        embedding = generate_embedding(question)
        print(f"Dimensión del embedding: {len(embedding)}")

        # Segunda etapa: consultar directamente Qdrant
        results = search_tasks(
            query_embedding=embedding,
            limit=10,
        )

        print("\nResultados directos de Qdrant:")

        for result in results:
            print(
                f"ID: {result.id} | "
                f"Score: {result.score:.4f} | "
                f"Payload: {result.payload}"
            )

        # Tercera etapa: consultar el recuperador completo
        tasks = retrieve_tasks(
            query=question,
            limit=10,
        )

        print("\nResultados de retrieve_tasks:")

        for task in tasks:
            print(f"ID: {task.id} | Nombre: {task.name}")

        if not tasks:
            print("No se recuperaron tareas.")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

from rag.vector_store import client, COLLECTION_NAME

print("\nTareas almacenadas en Excel:")
for task in read_excel():
    print(f"ID: {task.id} | Nombre: {task.name}")

print("\nContenido de Qdrant:")
points, next_offset = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=100,
    with_payload=True,
    with_vectors=False,
)

for point in points:
    print(
        f"ID: {point.id} | "
        f"Payload: {point.payload}"
    )

print(f"\nTotal de puntos en Qdrant: {len(points)}")