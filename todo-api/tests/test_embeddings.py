import unittest
from typing import Any
from unittest.mock import patch

import httpx

from rag import embeddings


class GenerateEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api_key_patcher = patch.object(embeddings, "API_KEY", "test-api-key")
        self.api_key_patcher.start()
        self.dimensions_patcher = patch.object(embeddings, "_embedding_dimensions", None)
        self.dimensions_patcher.start()
        self.addCleanup(self.dimensions_patcher.stop)
        self.addCleanup(self.api_key_patcher.stop)

    def response(self, status_code: int, json: Any) -> httpx.Response:
        return httpx.Response(
            status_code,
            json=json,
            request=httpx.Request("POST", embeddings.URL),
        )

    def test_returns_embedding_and_sends_bearer_auth(self) -> None:
        with patch.object(
            embeddings.httpx,
            "post",
            return_value=self.response(200, {"data": [{"embedding": [0.1, 0.2]}]}),
        ) as post:
            result = embeddings.generate_embedding("test text")

        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-api-key")

    def test_reports_connection_failures(self) -> None:
        with patch.object(
            embeddings.httpx,
            "post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with self.assertRaisesRegex(embeddings.EmbeddingError, "conectar"):
                embeddings.generate_embedding("test text")

    def test_reports_http_errors_without_exposing_response_body(self) -> None:
        response = self.response(401, {"error": "secret provider detail"})
        with patch.object(embeddings.httpx, "post", return_value=response):
            with self.assertRaisesRegex(embeddings.EmbeddingError, "HTTP \\(401\\)") as error:
                embeddings.generate_embedding("test text")

        self.assertNotIn("secret provider detail", str(error.exception))

    def test_rejects_unexpected_response_shapes(self) -> None:
        invalid_responses: list[Any] = [
            [],
            {},
            {"data": []},
            {"data": [{"embedding": []}]},
            {"data": [{"embedding": [0.1, "invalid"]}]},
        ]

        for body in invalid_responses:
            with self.subTest(body=body):
                with patch.object(
                    embeddings.httpx,
                    "post",
                    return_value=self.response(200, body),
                ):
                    with self.assertRaises(embeddings.EmbeddingError):
                        embeddings.generate_embedding("test text")

        response = self.response(200, {})
        with patch.object(
            response,
            "json",
            return_value={"data": [{"embedding": [float("nan")]}]},
        ), patch.object(embeddings.httpx, "post", return_value=response):
            with self.assertRaisesRegex(embeddings.EmbeddingError, "no finitos"):
                embeddings.generate_embedding("test text")

    def test_rejects_dimension_changes(self) -> None:
        responses = [
            self.response(200, {"data": [{"embedding": [0.1, 0.2]}]}),
            self.response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}),
        ]
        with patch.object(embeddings.httpx, "post", side_effect=responses):
            self.assertEqual(embeddings.generate_embedding("first"), [0.1, 0.2])
            with self.assertRaisesRegex(embeddings.EmbeddingError, "dimensión"):
                embeddings.generate_embedding("second")


if __name__ == "__main__":
    unittest.main()
