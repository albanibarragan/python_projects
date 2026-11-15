
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from schemas import TaskRecord
from rag.embeddings import EmbeddingError

import routers.llm as llm_router


@pytest.fixture(scope="module")
def client():
    with patch.object(main, "reconcile_tasks"), TestClient(app) as test_client:
        yield test_client


def test_chat_success(client, monkeypatch):
    task = TaskRecord(
        id=3,
        name="Estudiar Python",
        description="Repasar FastAPI y embeddings",
        date=date(2026, 11, 9),
        status="In progress",
    )

    def fake_retrieve_tasks(query, limit, status):
        return [task]

    def fake_generate_answer(question, tasks):
        return "Debes repasar FastAPI y embeddings."

    monkeypatch.setattr(
        llm_router,
        "retrieve_tasks",
        fake_retrieve_tasks,
    )

    monkeypatch.setattr(
        llm_router,
        "generate_answer",
        fake_generate_answer,
    )

    response = client.post(
        "/chat",
        json={
            "question": "¿Qué debo repasar de Python?",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Debes repasar FastAPI y embeddings."
    }


def test_chat_without_tasks(client, monkeypatch):
    def fake_retrieve_tasks(query, limit, status):
        return []

    monkeypatch.setattr(
        llm_router,
        "retrieve_tasks",
        fake_retrieve_tasks,
    )

    response = client.post(
        "/chat",
        json={
            "question": "¿Qué tareas tengo relacionadas con astronomía?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "No encontré tareas relacionadas con tu consulta."
    }


def test_chat_embedding_error(client, monkeypatch):
    def fake_retrieve_tasks(query, limit, status):
        raise EmbeddingError("Error simulado")

    monkeypatch.setattr(
        llm_router,
        "retrieve_tasks",
        fake_retrieve_tasks,
    )

    response = client.post(
        "/chat",
        json={
            "question": "¿Qué tareas tengo pendientes?",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "No se pudo procesar la consulta con el servicio de IA"
    }
