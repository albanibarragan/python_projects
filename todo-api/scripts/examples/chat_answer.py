
from rag.retriever import retrieve_tasks
from rag.generator import generate_answer

question = "¿Qué tarea tengo relacionada con Python y qué debo repasar?"

# Recuperar las tareas relevantes
tasks = retrieve_tasks(
    query=question,
    limit=3,
)

# Generar una respuesta utilizando esas tareas
answer = generate_answer(
    question=question,
    tasks=tasks,
)

print("Pregunta:", question)
print("\nRespuesta del asistente:")
print(answer)
