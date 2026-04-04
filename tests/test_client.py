"""
Integration tests for :class:`LLMClient`.

All HTTP calls are intercepted via ``unittest.mock.patch`` — no live
network traffic is produced by any test in this module.

Test classes
------------
* ``TestLLMClientFactory``     — ``LLMClient.create`` factory method
* ``TestLLMClientGenerate``    — synchronous ``generate()``
* ``TestLLMClientGenerateAsync`` — async ``generate_async()``
* ``TestBuildUrl``             — ``_build_url()`` URL construction
* ``TestBuildHeaders``         — ``_build_headers()`` header construction
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest  # type: ignore[import-untyped]

from apr_modules.llm_client import LLMClient  # type: ignore[import]
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
