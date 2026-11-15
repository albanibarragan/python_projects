import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")

url = f"{BASE_URL}/embeddings"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "nvidia/nemotron-3.5-embed-1b:free",
    "input": "Comprar comida en el supermercado",
}

response = httpx.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

print("Status:", response.status_code)
print(response.json())