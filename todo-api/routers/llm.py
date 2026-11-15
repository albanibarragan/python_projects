
import logging

from fastapi import APIRouter, HTTPException

from schemas import ChatRequest, ChatResponse

from rag.retriever import retrieve_tasks
from rag.generator import generate_answer, GenerationError
from rag.embeddings import EmbeddingError


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:

    try:
        # 1. Recuperar tareas relevantes
        tasks = retrieve_tasks(
            query=request.question,
            limit=request.limit,
            status=request.status,
        )

        # 2. Generar una respuesta con el contexto recuperado
        answer = generate_answer(
            question=request.question,
            tasks=tasks,
        )

        # 3. Devolver la respuesta
        return ChatResponse(answer=answer)

    except (EmbeddingError, GenerationError) as exc:
        logger.exception("Error al procesar la consulta RAG")

        raise HTTPException(
            status_code=502,
            detail="No se pudo procesar la consulta con el servicio de IA",
        ) from exc

    except Exception as exc:
        logger.exception("Error interno al procesar la consulta")

        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar la consulta",
        ) from exc
