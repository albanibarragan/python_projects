import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from routers.tasks import router as task_router
from routers.llm import router as llm_router
from rag.synchronizer import reconcile_tasks

logger = logging.getLogger(__name__)
RECONCILIATION_INTERVAL_SECONDS = 5 * 60


async def _reconcile_periodically() -> None:
    while True:
        await asyncio.sleep(RECONCILIATION_INTERVAL_SECONDS)
        try:
            await run_in_threadpool(reconcile_tasks)
        except Exception:
            logger.exception("Falló la reconciliación periódica Excel-Qdrant")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await run_in_threadpool(reconcile_tasks)
    except Exception:
        logger.exception(
            "No se pudo reconciliar Excel con Qdrant al iniciar; "
            "la API continuará funcionando"
        )

    reconciliation_task = asyncio.create_task(_reconcile_periodically())
    try:
        yield
    finally:
        reconciliation_task.cancel()
        try:
            await reconciliation_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="To Do List", lifespan=lifespan)
app.include_router(task_router)
app.include_router(llm_router)

@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Bienvenido a mi To-Do API"}


@app.get("/about")
def about() -> dict[str, str]:
    return {
        "project": "To-Do API",
        "developer": "albanibarragan",
        "status": "In development",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
