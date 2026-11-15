# To-Do API

API sencilla para crear, consultar, actualizar y eliminar tareas. Guarda las tareas en `tasks.xlsx`.
Excel es la fuente de verdad. Qdrant solo conserva el índice semántico; si una
operación de indexación falla, la reconciliación vuelve a intentarlo después.

## Búsqueda semántica con Qdrant

El módulo `rag/vector_store.py` guarda los embeddings de las tareas en una colección
llamada `tasks`. Por defecto usa persistencia local en `qdrant_db/`, cuya ubicación
puedes cambiar con `QDRANT_PATH`. Para conectarlo a un servidor Qdrant, configura
`QDRANT_URL` y, si el servidor lo requiere, `QDRANT_API_KEY` en `.env`; esa opción
tiene prioridad sobre el almacenamiento local. El modo local está pensado para
desarrollo y no debe compartirse entre varios procesos.

El almacén valida que los vectores coincidan con la dimensión de la colección.
`update_task_vector` actualiza un vector usando `upsert`, y `search_tasks` acepta el
parámetro opcional `status` para filtrar resultados por estado.

Instala las dependencias del proyecto y ejecuta el ejemplo de búsqueda:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m scripts.examples.vector_store_demo
```

El ejemplo genera embeddings usando OpenRouter; configura `OPENROUTER_API_KEY` y
`OPENROUTER_BASE_URL` en `.env` antes de ejecutarlo. `OPENROUTER_EMBEDDING_DIMENSIONS`
es opcional; si se configura, cada embedding debe tener exactamente ese número de
dimensiones. Si se omite, el proceso fija la dimensión al primer embedding válido.
La API de tareas existente continúa usando Excel como almacenamiento de registros.

## Estructura del proyecto

```text
todo-api/
├── main.py, schemas.py, storage.py   # Aplicación y persistencia Excel
├── routers/                          # Endpoints HTTP
├── rag/                              # Embeddings, recuperación y Qdrant
├── tests/                            # Suite automatizada de pytest
├── scripts/
│   ├── diagnostics/                  # Diagnóstico de RAG y Qdrant
│   ├── examples/                     # Ejemplos manuales de funcionalidades
│   └── maintenance/                  # Reindexación y mantenimiento
├── tasks.xlsx                        # Datos fuente (local, ignorado por Git)
└── qdrant_db/                        # Índice local (runtime, ignorado por Git)
```

Los scripts se ejecutan como módulos desde la raíz del repositorio. Algunos
ejemplos requieren `OPENROUTER_API_KEY`; los diagnósticos operan sobre la
configuración y los datos definidos en `.env`.

```powershell
# Diagnósticos
.\venv\Scripts\python.exe -m scripts.diagnostics.qdrant_status
.\venv\Scripts\python.exe -m scripts.diagnostics.debug_retrieval
.\venv\Scripts\python.exe -m scripts.diagnostics.retrieval_quality

# Mantenimiento
.\venv\Scripts\python.exe -m scripts.maintenance.reindex_tasks

# Suite automatizada
.\venv\Scripts\python.exe -m pytest -v
```

Al iniciar la API y cada cinco minutos mientras permanece activa, se comparan las
tareas de Excel con los documentos guardados en Qdrant. Se indexan las tareas
faltantes, se actualizan las que cambiaron y se eliminan los vectores de tareas
que ya no existen en Excel. Si falla la generación de un embedding o la conexión
con Qdrant, el error queda registrado y la aplicación vuelve a intentar la
reconciliación en el siguiente ciclo, sin detener la API.

El ciclo se ejecuta en segundo plano dentro del lifespan de FastAPI, espera a que
cada reconciliación termine antes de programar la siguiente y se cancela al apagar
la aplicación.

## Validación de fallos y recuperación

Las siguientes respuestas describen el comportamiento actual y están cubiertas
por pruebas que simulan fallos temporales:

- **¿Qué sucede si OpenRouter no responde durante una actualización?** La tarea
  primero se guarda en Excel. Si falla la generación del embedding, se registra el
  error y el endpoint PUT/PATCH sigue respondiendo con la tarea actualizada; el
  vector de Qdrant puede quedar temporalmente desactualizado.
- **¿Qué sucede si Qdrant falla durante la sincronización?** Los errores al indexar
  una tarea se registran y no detienen el resto de la reconciliación. Si falla la
  lectura inicial de Qdrant, el error se registra en el arranque y la API continúa
  disponible.
- **¿Puede recuperarse?** Sí. La sincronización compara Excel y Qdrant al iniciar
  y cada cinco minutos; intenta reparar los registros que falten o estén
  desactualizados. Las pruebas de resiliencia simulan un fallo de escritura en
  Qdrant y verifican tanto el reintento periódico como la reparación en el
  siguiente arranque.
- **¿Se mantienen los datos de Excel si falla la indexación?** Sí. Las operaciones
  de escritura en Excel ocurren antes de indexar; un error posterior de OpenRouter
  o Qdrant no revierte ni elimina el cambio guardado.

Ejecuta estas pruebas con:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_tasks.py tests\test_rag_resilience.py -v
```

La suite simula OpenRouter/Qdrant sin usar servicios externos y utiliza archivos
temporales para no modificar `tasks.xlsx` ni la colección real.

## Ejecutar

```powershell
.\venv\Scripts\fastapi.exe dev main.py
```
