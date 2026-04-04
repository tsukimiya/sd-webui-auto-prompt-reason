"""
Integration tests for :class:`LLMClient`.

All HTTP calls are intercepted via ``unittest.mock.patch`` — no live
network traffic is produced by any test in this module.

Test classes
------------
* ``TestLLMClientFactory``       — ``LLMClient.create`` factory method
* ``TestLLMClientGenerate``      — synchronous ``generate()``
* ``TestLLMClientGenerateAsync`` — async ``generate_async()``
* ``TestBuildUrl``               — ``_build_url()`` URL construction
* ``TestBuildHeaders``           — ``_build_headers()`` header construction
* ``TestRedactionHelpers``       — module-level redaction/summary helpers
* ``TestLogging``                — trace log messages emitted by ``generate()``
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest  # type: ignore[import-untyped]

from apr_modules.llm_client import LLMClient  # type: ignore[import]
from apr_modules.llm_client import (  # type: ignore[import]
    _redact_url,
    _redact_headers,
    _summarise_payload,
    _payload_meta,
    _contains_image,
)
from apr_modules.models.reasoning_response import ReasoningResponse  # type: ignore[import]
from apr_modules.providers.base_provider import SecretStr, ProviderType  # type: ignore[import]
from apr_modules.providers.ollama_provider import OllamaProvider  # type: ignore[import]
from apr_modules.providers.openai_provider import OpenAICompatibleProvider  # type: ignore[import]
from apr_modules.providers.gemini_provider import GeminiProvider  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(
    content: str = "answer",
    thinking: str = "thought",
    eval_count: int = 20,
    prompt_eval_count: int = 80,
) -> MagicMock:
    """Return a mock ``requests.Response`` that simulates an Ollama-shaped reply."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
        },
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
    }
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_ollama_client(model_name: str = "qwen3.5:7b") -> LLMClient:
    """Return a real :class:`LLMClient` backed by an :class:`OllamaProvider`."""
    return LLMClient.create("ollama", model_name=model_name)


def _make_openai_client(base_url: str = "http://localhost:1234/v1") -> LLMClient:
    """Return a real :class:`LLMClient` backed by :class:`OpenAICompatibleProvider`."""
    return LLMClient.create(
        "openai_compatible",
        base_url=base_url,
        model_name="mistral-7b",
        api_key=SecretStr("sk-test"),
    )


def _make_gemini_client() -> LLMClient:
    """Return a real :class:`LLMClient` backed by :class:`GeminiProvider`."""
    return LLMClient.create("gemini", api_key=SecretStr("gemini-api-key"))


# ===========================================================================
# TestLLMClientFactory
# ===========================================================================


class TestLLMClientFactory:
    """Tests for the :py:meth:`LLMClient.create` factory."""

    def test_create_ollama_returns_llm_client(self) -> None:
        """``create("ollama")`` returns an :class:`LLMClient` instance."""
        client = LLMClient.create("ollama", model_name="llama3.2:latest")
        assert isinstance(client, LLMClient)

    def test_create_ollama_provider_type(self) -> None:
        """Provider inside the returned client is :class:`OllamaProvider`."""
        client = LLMClient.create("ollama", model_name="llama3.2:latest")
        assert isinstance(client._provider, OllamaProvider)

    def test_create_openai_compatible_returns_llm_client(self) -> None:
        """``create("openai_compatible")`` returns an :class:`LLMClient` instance."""
        client = LLMClient.create(
            "openai_compatible",
            base_url="http://localhost:1234/v1",
            model_name="mistral-7b",
        )
        assert isinstance(client, LLMClient)

    def test_create_openai_provider_type(self) -> None:
        """Provider inside the returned client is :class:`OpenAICompatibleProvider`."""
        client = LLMClient.create(
            "openai_compatible",
            base_url="http://localhost:1234/v1",
            model_name="mistral-7b",
        )
        assert isinstance(client._provider, OpenAICompatibleProvider)

    def test_create_gemini_returns_llm_client(self) -> None:
        """``create("gemini", api_key=...)`` returns an :class:`LLMClient` instance."""
        client = LLMClient.create("gemini", api_key=SecretStr("test-key"))
        assert isinstance(client, LLMClient)

    def test_create_gemini_provider_type(self) -> None:
        """Provider inside the returned client is :class:`GeminiProvider`."""
        client = LLMClient.create("gemini", api_key=SecretStr("test-key"))
        assert isinstance(client._provider, GeminiProvider)

    def test_create_unknown_provider_raises_value_error(self) -> None:
        """Unknown provider string raises :py:exc:`ValueError`."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            LLMClient.create("anthropic")

    def test_create_unknown_provider_message_contains_name(self) -> None:
        """Error message for unknown provider includes the bad string."""
        with pytest.raises(ValueError, match="'does_not_exist'"):
            LLMClient.create("does_not_exist")


# ===========================================================================
# TestLLMClientGenerate
# ===========================================================================


class TestLLMClientGenerate:
    """Tests for :py:meth:`LLMClient.generate`."""

    def test_generate_returns_reasoning_response(self) -> None:
        """Successful call returns a :class:`ReasoningResponse`."""
        client = _make_ollama_client()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            result = client.generate("test prompt")
        assert isinstance(result, ReasoningResponse)

    def test_generate_reasoning_time_ms_is_positive_int(self) -> None:
        """``reasoning_time_ms`` on the returned response is a positive integer."""
        client = _make_ollama_client()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            result = client.generate("test prompt")
        assert isinstance(result.reasoning_time_ms, int)
        assert result.reasoning_time_ms >= 0

    def test_generate_calls_validate_config(self) -> None:
        """``generate()`` calls ``validate_config`` on the provider."""
        mock_provider = MagicMock(spec=OllamaProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OLLAMA
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.timeout = (10, 60)
        mock_provider.model_name = "qwen3.5:7b"
        mock_provider.api_key = None
        mock_provider.build_request.return_value = {"model": "qwen3.5:7b", "messages": []}
        mock_provider.parse_response.return_value = ReasoningResponse(
            final_answer="answer",
            thinking_content="thought",
            reasoning_tokens=5,
            completion_tokens=20,
            total_tokens=100,
            reasoning_time_ms=0,
            model="qwen3.5:7b",
            provider="ollama",
            raw_response={},
        )
        client = LLMClient(mock_provider)
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            client.generate("test prompt")
        mock_provider.validate_config.assert_called_once()

    def test_generate_http_error_propagates(self) -> None:
        """HTTP 500 from the server propagates as ``requests.HTTPError``."""
        import requests as req_module

        client = _make_ollama_client()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req_module.HTTPError("500 Server Error")
            mock_post.return_value = mock_resp
            with pytest.raises(req_module.HTTPError):
                client.generate("test prompt")

    def test_generate_reasoning_effort_passed_to_build_request(self) -> None:
        """``reasoning_effort`` argument is forwarded to ``build_request``."""
        mock_provider = MagicMock(spec=OllamaProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OLLAMA
        mock_provider.base_url = "http://localhost:11434"
        mock_provider.timeout = (10, 60)
        mock_provider.model_name = "qwen3.5:7b"
        mock_provider.api_key = None
        mock_provider.build_request.return_value = {"model": "qwen3.5:7b", "messages": []}
        mock_provider.parse_response.return_value = ReasoningResponse(
            final_answer="answer",
            thinking_content=None,
            reasoning_tokens=None,
            completion_tokens=20,
            total_tokens=100,
            reasoning_time_ms=0,
            model="qwen3.5:7b",
            provider="ollama",
            raw_response={},
        )
        client = LLMClient(mock_provider)
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            client.generate("test prompt", reasoning_effort="high")
        mock_provider.build_request.assert_called_once_with("test prompt", None, "high", None)

    def test_generate_final_answer_matches_provider_parse(self) -> None:
        """The ``final_answer`` in the result matches what the provider parses."""
        client = _make_ollama_client()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response(content="a cat on a bench")
            result = client.generate("describe this")
        assert result.final_answer == "a cat on a bench"

    def test_generate_no_live_http(self) -> None:
        """No live HTTP calls occur when ``requests.post`` is mocked."""
        client = _make_ollama_client()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            client.generate("test prompt")
            assert mock_post.called
            # Only one POST call should have been made.
            assert mock_post.call_count == 1


# ===========================================================================
# TestLLMClientGenerateAsync
# ===========================================================================


class TestLLMClientGenerateAsync:
    """Tests for :py:meth:`LLMClient.generate_async`."""

    def test_generate_async_returns_thread(self) -> None:
        """``generate_async`` returns a :class:`threading.Thread`."""
        client = _make_ollama_client()
        callback = MagicMock()
        error_callback = MagicMock()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            thread = client.generate_async("prompt", callback, error_callback)
        thread.join(timeout=5)
        assert isinstance(thread, threading.Thread)

    def test_generate_async_thread_is_daemon(self) -> None:
        """The returned thread is a daemon thread."""
        client = _make_ollama_client()
        callback = MagicMock()
        error_callback = MagicMock()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            thread = client.generate_async("prompt", callback, error_callback)
        assert thread.daemon is True
        thread.join(timeout=5)

    def test_generate_async_callback_called_with_reasoning_response(self) -> None:
        """On success, ``callback`` is invoked with a :class:`ReasoningResponse`."""
        client = _make_ollama_client()
        received: list[ReasoningResponse] = []
        error_received: list[Exception] = []

        def on_success(r: ReasoningResponse) -> None:
            received.append(r)

        def on_error(e: Exception) -> None:
            error_received.append(e)

        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            thread = client.generate_async("test prompt", on_success, on_error)

        thread.join(timeout=5)
        assert len(received) == 1
        assert isinstance(received[0], ReasoningResponse)
        assert len(error_received) == 0

    def test_generate_async_error_callback_called_on_exception(self) -> None:
        """When ``generate()`` raises, ``error_callback`` is called with the exception."""
        import requests as req_module

        client = _make_ollama_client()
        received: list[ReasoningResponse] = []
        errors: list[Exception] = []

        def on_success(r: ReasoningResponse) -> None:
            received.append(r)

        def on_error(e: Exception) -> None:
            errors.append(e)

        with patch("modules.llm_client.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req_module.HTTPError("500")
            mock_post.return_value = mock_resp
            thread = client.generate_async("test prompt", on_success, on_error)

        thread.join(timeout=5)
        assert len(errors) == 1
        assert isinstance(errors[0], req_module.HTTPError)
        assert len(received) == 0

    def test_generate_async_thread_completes(self) -> None:
        """The worker thread finishes within a reasonable timeout."""
        client = _make_ollama_client()
        callback = MagicMock()
        error_callback = MagicMock()
        with patch("modules.llm_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response()
            thread = client.generate_async("test prompt", callback, error_callback)
        thread.join(timeout=5)
        assert not thread.is_alive()


# ===========================================================================
# TestBuildUrl
# ===========================================================================


class TestBuildUrl:
    """Tests for :py:meth:`LLMClient._build_url`."""

    def test_ollama_url_ends_with_api_chat(self) -> None:
        """Ollama URL ends with ``/api/chat``."""
        client = _make_ollama_client()
        url = client._build_url()
        assert url.endswith("/api/chat")

    def test_ollama_url_contains_base_url(self) -> None:
        """Ollama URL starts with the configured base URL."""
        client = LLMClient.create(
            "ollama",
            base_url="http://localhost:11434",
            model_name="llama3.2:latest",
        )
        url = client._build_url()
        assert url == "http://localhost:11434/api/chat"

    def test_openai_with_v1_ends_with_chat_completions(self) -> None:
        """OpenAI base URL that already includes ``/v1`` → ``/chat/completions``."""
        client = _make_openai_client(base_url="http://localhost:1234/v1")
        url = client._build_url()
        assert url.endswith("/chat/completions")
        assert url == "http://localhost:1234/v1/chat/completions"

    def test_openai_without_v1_appends_v1_chat_completions(self) -> None:
        """OpenAI base URL without ``/v1`` gets ``/v1/chat/completions`` appended."""
        client = _make_openai_client(base_url="http://localhost:1234")
        url = client._build_url()
        assert "/v1/chat/completions" in url
        assert url == "http://localhost:1234/v1/chat/completions"

    def test_gemini_url_contains_key_query_param(self) -> None:
        """Gemini URL contains ``?key=`` query parameter."""
        client = _make_gemini_client()
        url = client._build_url()
        assert "?key=" in url

    def test_gemini_url_contains_generate_content(self) -> None:
        """Gemini URL contains ``:generateContent``."""
        client = _make_gemini_client()
        url = client._build_url()
        assert "generateContent" in url

    def test_gemini_url_embeds_api_key_value(self) -> None:
        """Gemini URL embeds the actual API key value in the query string."""
        client = LLMClient.create("gemini", api_key=SecretStr("my-secret-key"))
        url = client._build_url()
        assert "my-secret-key" in url


# ===========================================================================
# TestBuildHeaders
# ===========================================================================


class TestBuildHeaders:
    """Tests for :py:meth:`LLMClient._build_headers`."""

    def test_all_providers_have_content_type(self) -> None:
        """Every provider returns ``Content-Type: application/json``."""
        for client in [_make_ollama_client(), _make_openai_client(), _make_gemini_client()]:
            headers = client._build_headers()
            assert headers.get("Content-Type") == "application/json"

    def test_ollama_without_api_key_no_authorization(self) -> None:
        """Ollama without ``api_key`` has no ``Authorization`` header."""
        client = LLMClient.create(
            "ollama",
            base_url="http://localhost:11434",
            model_name="qwen3.5:7b",
            api_key=None,
        )
        headers = client._build_headers()
        assert "Authorization" not in headers

    def test_ollama_with_api_key_has_authorization(self) -> None:
        """Ollama with an ``api_key`` includes ``Authorization: Bearer …``."""
        client = LLMClient.create(
            "ollama",
            base_url="http://localhost:11434",
            model_name="qwen3.5:7b",
            api_key=SecretStr("ollama-token"),
        )
        headers = client._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer ollama-token"

    def test_openai_with_api_key_has_authorization(self) -> None:
        """OpenAI-compatible with ``api_key`` includes ``Authorization: Bearer …``."""
        client = _make_openai_client()
        headers = client._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test"

    def test_gemini_no_authorization_header(self) -> None:
        """Gemini provider never adds an ``Authorization`` header."""
        client = _make_gemini_client()
        headers = client._build_headers()
        assert "Authorization" not in headers

    def test_gemini_with_api_key_still_no_authorization_header(self) -> None:
        """Even with an API key, Gemini omits ``Authorization`` (uses ``?key=`` in URL)."""
        client = LLMClient.create("gemini", api_key=SecretStr("gemini-key"))
        headers = client._build_headers()
        assert "Authorization" not in headers


# ===========================================================================
# TestLLMClientSystemPromptForwarding
# ===========================================================================


class TestLLMClientSystemPromptForwarding:
    """Tests that ``system_prompt`` is accepted by provider ``build_request`` methods.

    These tests use the real provider objects (accessed through the client's
    ``_provider``) to verify that the ``system_prompt`` kwarg produces the
    correct payload shape for each backend.
    """

    # --- Ollama ---

    def test_ollama_build_request_system_prompt_via_provider(self) -> None:
        """Ollama provider inside client accepts system_prompt in build_request."""
        client = LLMClient.create("ollama", model_name="llama3.2:latest")
        payload = client._provider.build_request(
            "describe this",
            system_prompt="You are an artist.",
        )
        assert payload["messages"][0] == {"role": "system", "content": "You are an artist."}
        assert payload["messages"][1]["role"] == "user"

    def test_ollama_build_request_no_system_prompt_via_provider(self) -> None:
        """Ollama provider inside client omits system message when system_prompt is None."""
        client = LLMClient.create("ollama", model_name="llama3.2:latest")
        payload = client._provider.build_request("describe this")
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    # --- OpenAI-compatible ---

    def test_openai_build_request_system_prompt_via_provider(self) -> None:
        """OpenAI provider inside client accepts system_prompt in build_request."""
        client = LLMClient.create(
            "openai_compatible",
            base_url="http://localhost:1234/v1",
            model_name="mistral-7b",
        )
        payload = client._provider.build_request(
            "describe this",
            system_prompt="Be brief.",
        )
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
        assert payload["messages"][1]["role"] == "user"

    def test_openai_build_request_no_system_prompt_via_provider(self) -> None:
        """OpenAI provider inside client omits system message when system_prompt is None."""
        client = LLMClient.create(
            "openai_compatible",
            base_url="http://localhost:1234/v1",
            model_name="mistral-7b",
        )
        payload = client._provider.build_request("describe this")
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    # --- Gemini ---

    def test_gemini_build_request_system_prompt_via_provider(self) -> None:
        """Gemini provider inside client accepts system_prompt in build_request."""
        client = LLMClient.create("gemini", api_key=SecretStr("key"))
        payload = client._provider.build_request(
            "describe this",
            system_prompt="Respond in JSON.",
        )
        assert "systemInstruction" in payload
        assert payload["systemInstruction"]["parts"][0]["text"] == "Respond in JSON."

    def test_gemini_build_request_no_system_prompt_via_provider(self) -> None:
        """Gemini provider inside client omits systemInstruction when system_prompt is None."""
        client = LLMClient.create("gemini", api_key=SecretStr("key"))
        payload = client._provider.build_request("describe this")
        assert "systemInstruction" not in payload


# ===========================================================================
# TestRedactionHelpers
# ===========================================================================


class TestRedactionHelpers:
    """Unit tests for the module-level redaction / summary helpers."""

    # --- _redact_url ---

    def test_redact_url_removes_key_param_value(self) -> None:
        """``?key=<secret>`` is replaced with ``?key=**redacted**``."""
        url = "https://api.example.com/v1/models/gemini-2.0-flash:generateContent?key=supersecret"
        redacted = _redact_url(url)
        assert "supersecret" not in redacted
        assert "?key=**redacted**" in redacted

    def test_redact_url_preserves_non_key_urls(self) -> None:
        """URLs without a ``?key=`` param are returned unchanged."""
        url = "http://localhost:11434/api/chat"
        assert _redact_url(url) == url

    def test_redact_url_preserves_path_before_key(self) -> None:
        """The URL path before ``?key=`` is unchanged."""
        url = "https://api.example.com/v1/models/foo:gen?key=abc123"
        redacted = _redact_url(url)
        assert "https://api.example.com/v1/models/foo:gen" in redacted

    def test_redact_url_does_not_expose_key_value(self) -> None:
        """After redaction, the original key value is not present anywhere."""
        secret = "my-very-secret-gemini-key"
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini:gen?key={secret}"
        assert secret not in _redact_url(url)

    # --- _redact_headers ---

    def test_redact_headers_masks_authorization(self) -> None:
        """``Authorization`` header value is replaced with ``Bearer **redacted**``."""
        headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-real-key"}
        result = _redact_headers(headers)
        assert result["Authorization"] == "Bearer **redacted**"
        assert "sk-real-key" not in str(result)

    def test_redact_headers_preserves_content_type(self) -> None:
        """Non-secret headers pass through unchanged."""
        headers = {"Content-Type": "application/json"}
        assert _redact_headers(headers) == {"Content-Type": "application/json"}

    def test_redact_headers_no_mutation(self) -> None:
        """Original headers dict is not mutated."""
        headers = {"Authorization": "Bearer real-secret"}
        _redact_headers(headers)
        assert headers["Authorization"] == "Bearer real-secret"

    def test_redact_headers_case_insensitive_authorization(self) -> None:
        """``authorization`` (lowercase) is also masked."""
        headers = {"authorization": "Bearer sk-lower"}
        result = _redact_headers(headers)
        assert "sk-lower" not in str(result)

    # --- _summarise_payload ---

    def test_summarise_payload_truncates_gemini_inline_data(self) -> None:
        """Gemini ``inlineData.data`` blobs are truncated, not dumped in full."""
        long_b64 = "A" * 500
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": "describe this"},
                        {"inlineData": {"mimeType": "image/jpeg", "data": long_b64}},
                    ]
                }
            ]
        }
        summary = _summarise_payload(payload)
        # Navigate to the data field in the summary.
        data_val = summary["contents"][0]["parts"][1]["inlineData"]["data"]
        assert len(data_val) < len(long_b64)
        assert "chars total" in data_val

    def test_summarise_payload_truncates_openai_data_url(self) -> None:
        """OpenAI-style ``data:image/…;base64,…`` URLs are truncated."""
        long_b64 = "B" * 500
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": f"data:image/jpeg;base64,{long_b64}"}
                    ],
                }
            ]
        }
        summary = _summarise_payload(payload)
        url_val = summary["messages"][0]["content"][0]["url"]
        assert len(url_val) < len(long_b64) + 30  # much shorter than original
        assert "chars total" in url_val

    def test_summarise_payload_does_not_mutate_original(self) -> None:
        """``_summarise_payload`` returns a deep copy; original is untouched."""
        long_b64 = "C" * 200
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": long_b64}}]}]
        }
        _summarise_payload(payload)
        assert payload["contents"][0]["parts"][0]["inlineData"]["data"] == long_b64

    def test_summarise_payload_text_only_unchanged(self) -> None:
        """Text-only payloads pass through without modification."""
        payload: dict[str, Any] = {
            "model": "qwen3.5:7b",
            "messages": [{"role": "user", "content": "hello"}],
        }
        summary = _summarise_payload(payload)
        assert summary["messages"][0]["content"] == "hello"

    # --- _payload_meta ---

    def test_payload_meta_reports_message_count(self) -> None:
        """``_payload_meta`` includes the message count for Ollama/OpenAI payloads."""
        payload: dict[str, Any] = {
            "model": "llama3",
            "messages": [{"role": "user"}, {"role": "assistant"}],
        }
        meta = _payload_meta(payload)
        assert "messages=2" in meta

    def test_payload_meta_detects_image(self) -> None:
        """``_payload_meta`` reports ``has_image=True`` when image data present."""
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": "abc123"}}
                    ]
                }
            ]
        }
        meta = _payload_meta(payload)
        assert "has_image=True" in meta

    def test_payload_meta_detects_system_instruction(self) -> None:
        """``_payload_meta`` reports ``has_system=True`` for Gemini system instructions."""
        payload: dict[str, Any] = {
            "contents": [],
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
        }
        meta = _payload_meta(payload)
        assert "has_system=True" in meta

    def test_payload_meta_no_image_no_system(self) -> None:
        """Minimal text payload has no image/system flags."""
        payload: dict[str, Any] = {"model": "test", "messages": [{"role": "user"}]}
        meta = _payload_meta(payload)
        assert "has_image" not in meta
        assert "has_system" not in meta

    # --- _contains_image ---

    def test_contains_image_true_for_inline_data(self) -> None:
        """Returns ``True`` when Gemini ``inlineData`` is present."""
        assert _contains_image({"inlineData": {"data": "abc"}}) is True

    def test_contains_image_true_for_openai_image_url(self) -> None:
        """Returns ``True`` for OpenAI ``type: image_url`` content part."""
        assert _contains_image({"type": "image_url", "url": "data:image/jpeg;base64,abc"}) is True

    def test_contains_image_false_for_text_only(self) -> None:
        """Returns ``False`` for text-only content."""
        assert _contains_image({"type": "text", "text": "hello"}) is False

    def test_contains_image_recursive_list(self) -> None:
        """Recursively detects images nested inside lists."""
        obj = [{"text": "hi"}, {"inlineData": {"data": "xyz"}}]
        assert _contains_image(obj) is True


# ===========================================================================
# TestLogging
# ===========================================================================

_PATCH_REQUESTS = "apr_modules.llm_client.requests.post"


def _make_mock_response_for_log_tests(
    content: str = "answer",
) -> MagicMock:
    """Return a minimal mock response for logging tests (Ollama-shaped)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": content, "thinking": ""},
        "eval_count": 10,
        "prompt_eval_count": 40,
    }
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestLogging:
    """Verify trace log messages emitted by :py:meth:`LLMClient.generate`.

    All tests capture DEBUG records from ``apr_modules.llm_client`` and check
    that the expected lifecycle stages are logged and that no secrets appear.
    """

    def _capture_logs(
        self,
        client: LLMClient,
        prompt: str = "test prompt",
        image_data: "str | None" = None,
        system_prompt: "str | None" = None,
        reasoning_effort: str = "medium",
    ) -> list[str]:
        """Run ``generate()`` with a mocked HTTP response and return log messages."""
        captured: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        logger = logging.getLogger("apr_modules.llm_client")
        handler = _Handler()
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            with patch(_PATCH_REQUESTS) as mock_post:
                mock_post.return_value = _make_mock_response_for_log_tests()
                client.generate(
                    prompt,
                    image_data=image_data,
                    system_prompt=system_prompt,
                    reasoning_effort=reasoning_effort,
                )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        return captured

    # --- lifecycle stages ---

    def test_config_validated_logged(self) -> None:
        """``[CONFIG VALIDATED]`` appears in the log on a successful call."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("[CONFIG VALIDATED]" in msg for msg in logs)

    def test_payload_prepared_logged(self) -> None:
        """``[PAYLOAD PREPARED]`` appears in the log on a successful call."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("[PAYLOAD PREPARED]" in msg for msg in logs)

    def test_request_sending_logged(self) -> None:
        """``[REQUEST SENDING]`` appears just before the HTTP call."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("[REQUEST SENDING]" in msg for msg in logs)

    def test_response_received_logged(self) -> None:
        """``[RESPONSE RECEIVED]`` appears after a successful HTTP response."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("[RESPONSE RECEIVED]" in msg for msg in logs)

    def test_json_parsed_logged(self) -> None:
        """``[JSON PARSED]`` appears after response JSON is decoded."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("[JSON PARSED]" in msg for msg in logs)

    def test_request_failed_logged_on_http_error(self) -> None:
        """``[REQUEST FAILED]`` is logged when the server returns an HTTP error."""
        import requests as req_module

        client = _make_ollama_client()
        captured: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        logger = logging.getLogger("apr_modules.llm_client")
        handler = _Handler()
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            with patch(_PATCH_REQUESTS) as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                mock_resp.raise_for_status.side_effect = req_module.HTTPError("500 Server Error")
                mock_post.return_value = mock_resp
                with pytest.raises(req_module.HTTPError):
                    client.generate("test prompt")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        assert any("[REQUEST FAILED]" in msg for msg in captured)

    def test_request_failed_logged_on_connection_error(self) -> None:
        """``[REQUEST FAILED]`` is logged for low-level network errors."""
        import requests as req_module

        client = _make_ollama_client()
        captured: list[str] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        logger = logging.getLogger("apr_modules.llm_client")
        handler = _Handler()
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            with patch(_PATCH_REQUESTS) as mock_post:
                mock_post.side_effect = req_module.ConnectionError("refused")
                with pytest.raises(req_module.ConnectionError):
                    client.generate("test prompt")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        assert any("[REQUEST FAILED]" in msg for msg in captured)

    # --- provider / model metadata in logs ---

    def test_provider_name_in_logs(self) -> None:
        """``provider=ollama`` appears in log output."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("provider=ollama" in msg for msg in logs)

    def test_model_name_in_logs(self) -> None:
        """Model name appears in log output."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        assert any("qwen3.5:7b" in msg for msg in logs)

    def test_elapsed_ms_in_response_received_log(self) -> None:
        """``elapsed_ms=`` field appears in the ``[RESPONSE RECEIVED]`` log."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        response_log = next(m for m in logs if "[RESPONSE RECEIVED]" in m)
        assert "elapsed_ms=" in response_log

    def test_http_status_in_response_received_log(self) -> None:
        """HTTP status code 200 appears in the ``[RESPONSE RECEIVED]`` log."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        response_log = next(m for m in logs if "[RESPONSE RECEIVED]" in m)
        assert "status=200" in response_log

    def test_response_keys_in_json_parsed_log(self) -> None:
        """Top-level response JSON keys appear in the ``[JSON PARSED]`` log."""
        client = _make_ollama_client()
        logs = self._capture_logs(client)
        json_log = next(m for m in logs if "[JSON PARSED]" in m)
        # The mock response has keys: message, eval_count, prompt_eval_count
        assert "message" in json_log

    def test_image_presence_flagged_in_payload_log(self) -> None:
        """``has_image=True`` appears in ``[PAYLOAD PREPARED]`` when image provided."""
        client = _make_ollama_client()
        logs = self._capture_logs(client, image_data="fake_base64_image_data")
        payload_log = next(m for m in logs if "[PAYLOAD PREPARED]" in m)
        assert "has_image=True" in payload_log

    def test_system_prompt_flagged_in_payload_log(self) -> None:
        """``has_system=True`` appears in ``[PAYLOAD PREPARED]`` when system_prompt provided."""
        client = _make_ollama_client()
        logs = self._capture_logs(client, system_prompt="You are helpful.")
        payload_log = next(m for m in logs if "[PAYLOAD PREPARED]" in m)
        assert "has_system=True" in payload_log

    # --- redaction / no-leak checks ---

    def test_openai_api_key_not_in_logs(self) -> None:
        """Raw OpenAI API key is never present in any log message."""
        client = _make_openai_client()  # uses api_key=SecretStr("sk-test")
        logs = self._capture_logs(client)
        for msg in logs:
            assert "sk-test" not in msg, f"API key leaked in log: {msg!r}"

    def test_gemini_api_key_not_in_request_sending_log(self) -> None:
        """Gemini ``?key=<value>`` is redacted in ``[REQUEST SENDING]`` log."""
        client = LLMClient.create("gemini", api_key=SecretStr("gemini-secret-key"))
        logs = self._capture_logs(client)
        sending_logs = [m for m in logs if "[REQUEST SENDING]" in m]
        assert sending_logs, "Expected at least one [REQUEST SENDING] log"
        for msg in sending_logs:
            assert "gemini-secret-key" not in msg, f"Gemini key leaked in log: {msg!r}"

    def test_authorization_header_value_not_in_logs(self) -> None:
        """Raw ``Authorization: Bearer <token>`` value is not logged."""
        client = _make_openai_client()  # has Authorization header with "sk-test"
        logs = self._capture_logs(client)
        for msg in logs:
            # The token value must not appear even if "Authorization" appears
            assert "Bearer sk-test" not in msg, f"Auth header leaked: {msg!r}"

    def test_async_worker_started_logged(self) -> None:
        """``[ASYNC WORKER STARTED]`` is logged when ``generate_async`` is called."""
        client = _make_ollama_client()
        captured: list[str] = []
        done = threading.Event()

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        logger = logging.getLogger("apr_modules.llm_client")
        handler = _Handler()
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        def on_done(_: object) -> None:
            done.set()

        try:
            with patch(_PATCH_REQUESTS) as mock_post:
                mock_post.return_value = _make_mock_response_for_log_tests()
                client.generate_async("prompt", on_done, on_done)
            done.wait(timeout=5)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        assert any("[ASYNC WORKER STARTED]" in m for m in captured)

    def test_async_worker_done_logged_on_success(self) -> None:
        """``[ASYNC WORKER DONE]`` is logged when ``generate_async`` succeeds."""
        client = _make_ollama_client()
        captured: list[str] = []
        done = threading.Event()

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        logger = logging.getLogger("apr_modules.llm_client")
        handler = _Handler()
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        def on_done(_: object) -> None:
            done.set()

        try:
            with patch(_PATCH_REQUESTS) as mock_post:
                mock_post.return_value = _make_mock_response_for_log_tests()
                client.generate_async("prompt", on_done, on_done)
            done.wait(timeout=5)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        assert any("[ASYNC WORKER DONE]" in m for m in captured)
