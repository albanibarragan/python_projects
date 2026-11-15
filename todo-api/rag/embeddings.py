
import math
import os
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import TypeGuard

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

URL = f"{BASE_URL.rstrip('/')}/embeddings"
MODEL = "nvidia/nemotron-3.5-embed-1b:free"
_configured_dimensions = os.getenv("OPENROUTER_EMBEDDING_DIMENSIONS")
if _configured_dimensions is None:
    _embedding_dimensions: int | None = None
else:
    try:
        _embedding_dimensions = int(_configured_dimensions)
    except ValueError as exc:
        raise ValueError("OPENROUTER_EMBEDDING_DIMENSIONS debe ser un entero positivo") from exc
    if _embedding_dimensions < 1:
        raise ValueError("OPENROUTER_EMBEDDING_DIMENSIONS debe ser un entero positivo")

_dimensions_lock = Lock()


class EmbeddingError(RuntimeError):
    """Error al solicitar o validar un embedding de OpenRouter."""


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def generate_embedding(text: str) -> list[float]:
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY no está configurada")

    if not text or not text.strip():
        raise ValueError("El texto no puede estar vacío")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": text,
    }

    try:
        response = httpx.post(
            URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except httpx.RequestError as exc:
        raise EmbeddingError("No se pudo conectar con OpenRouter") from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(
            f"OpenRouter devolvió un error HTTP ({exc.response.status_code})"
        ) from exc

    try:
        response_data: object = response.json()
    except ValueError as exc:
        raise EmbeddingError("OpenRouter devolvió una respuesta que no es JSON válido") from exc

    if not _is_json_object(response_data):
        raise EmbeddingError("La respuesta de OpenRouter tiene un formato inesperado")

    items = response_data.get("data")
    if not _is_json_array(items) or not items or not _is_json_object(items[0]):
        raise EmbeddingError("La respuesta de OpenRouter no contiene un embedding")

    raw_embedding = items[0].get("embedding")
    if not _is_json_array(raw_embedding) or not raw_embedding:
        raise EmbeddingError("OpenRouter devolvió un embedding vacío o inválido")

    embedding: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingError("El embedding contiene valores no numéricos o no finitos")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise EmbeddingError("El embedding contiene valores no numéricos o no finitos")
        embedding.append(numeric_value)
    actual_dimensions = len(embedding)
    global _embedding_dimensions
    with _dimensions_lock:
        if _embedding_dimensions is None:
            _embedding_dimensions = actual_dimensions
        elif actual_dimensions != _embedding_dimensions:
            raise EmbeddingError(
                "La dimensión del embedding no coincide "
                f"(esperada: {_embedding_dimensions}, recibida: {actual_dimensions})"
            )

    return embedding
