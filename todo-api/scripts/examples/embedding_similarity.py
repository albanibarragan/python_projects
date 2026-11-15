import os
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")

URL = f"{BASE_URL}/embeddings"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def generate_embedding(text: str) -> list[float]:
    payload = {
        "model": "nvidia/nemotron-3.5-embed-1b:free",
        "input": text,
    }

    response = httpx.post(
        URL,
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def cosine_similarity(vector_a, vector_b) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)

    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )

reference = "Aprender Python para desarrollar aplicaciones backend"

texts = [
    "Estudiar programación con Python",
    "Aprender a cocinar comida italiana",
    "Desarrollar una API REST con FastAPI",
]

reference_embedding = generate_embedding(reference)

results = []

for text in texts:
    embedding = generate_embedding(text)

    similarity = cosine_similarity(
        reference_embedding,
        embedding,
    )

    results.append({
        "text": text,
        "similarity": similarity,
    })

results.sort(
    key=lambda item: item["similarity"],
    reverse=True,
)

for result in results:
    print(f"{result['similarity']:.4f} - {result['text']}")