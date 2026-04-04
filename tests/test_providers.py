"""
Comprehensive unit tests for all three provider implementations.

Tests cover:
- OllamaProvider
- OpenAICompatibleProvider
- GeminiProvider
- SecretStr

All tests use pure method calls — no live HTTP requests.
"""

from __future__ import annotations

from typing import Any

import pytest  # type: ignore[import-untyped]

from modules.providers.base_provider import ProviderType, SecretStr  # type: ignore[import]
from modules.providers.ollama_provider import OllamaProvider  # type: ignore[import]
from modules.providers.openai_provider import OpenAICompatibleProvider  # type: ignore[import]
from modules.providers.gemini_provider import GeminiProvider  # type: ignore[import]


# ===========================================================================
# OllamaProvider tests
# ===========================================================================


class TestOllamaProvider:
    """Unit tests for :class:`OllamaProvider`."""

    def _make_provider(self, model_name: str = "qwen3.5:7b") -> OllamaProvider:
        return OllamaProvider(
            base_url="http://localhost:11434",
            model_name=model_name,
        )

    # --- Identity & capability ---

    def test_ollama_get_provider_type(self) -> None:
        """get_provider_type() returns ProviderType.OLLAMA."""
        provider = self._make_provider()
        assert provider.get_provider_type() == ProviderType.OLLAMA

    def test_ollama_supports_reasoning(self) -> None:
        """supports_reasoning() returns True."""
        provider = self._make_provider()
        assert provider.supports_reasoning() is True

    # --- validate_config ---

    def test_ollama_validate_config_empty_model_raises(self) -> None:
        """Empty model_name raises ValueError."""
        provider = OllamaProvider(model_name="")
        with pytest.raises(ValueError, match="model_name"):
            provider.validate_config()

    def test_ollama_validate_config_ok(self) -> None:
        """Non-empty model_name does not raise."""
        provider = self._make_provider(model_name="llama3.2:latest")
        provider.validate_config()  # should not raise

    # --- build_request ---

    def test_ollama_build_request_basic(self) -> None:
        """build_request() produces the expected payload structure."""
        provider = self._make_provider(model_name="llama3.2:latest")
        payload = provider.build_request("describe this image")

        assert payload["model"] == "llama3.2:latest"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "describe this image"
        assert "images" not in payload["messages"][0]

    def test_ollama_build_request_with_image(self) -> None:
        """When image_data is provided, it appears in the messages images field."""
        provider = self._make_provider()
        payload = provider.build_request("describe this", image_data="base64abc==")

        assert "images" in payload["messages"][0]
        assert payload["messages"][0]["images"] == ["base64abc=="]

    # --- effort → temperature mapping ---

    def test_ollama_effort_temperature_mapping(self) -> None:
        """low=0.8, medium=0.5, high=0.2 temperature values."""
        provider = self._make_provider()

        low_payload = provider.build_request("x", reasoning_effort="low")
        medium_payload = provider.build_request("x", reasoning_effort="medium")
        high_payload = provider.build_request("x", reasoning_effort="high")

        assert low_payload["options"]["temperature"] == 0.8
        assert medium_payload["options"]["temperature"] == 0.5
        assert high_payload["options"]["temperature"] == 0.2

    # --- parse_response ---

    def test_ollama_parse_response_shape_a(
        self, ollama_raw_shape_a: dict[str, Any]
    ) -> None:
        """Shape A: dedicated thinking field is parsed correctly."""
        provider = self._make_provider()
        result = provider.parse_response(ollama_raw_shape_a)

        assert result.thinking_content == "Let me think"
        assert result.final_answer == "a cat sitting"

    def test_ollama_parse_response_shape_b(
        self, ollama_raw_shape_b: dict[str, Any]
    ) -> None:
        """Shape B: <think> tags are extracted into thinking_content."""
        provider = self._make_provider()
        result = provider.parse_response(ollama_raw_shape_b)

        assert result.thinking_content == "Let me think"
        assert result.final_answer == "a cat sitting"

    def test_ollama_parse_response_token_counts(
        self, ollama_raw_shape_a: dict[str, Any]
    ) -> None:
        """eval_count and prompt_eval_count are mapped to token fields."""
        provider = self._make_provider()
        result = provider.parse_response(ollama_raw_shape_a)

        # eval_count=20, prompt_eval_count=80 → total=100
        assert result.completion_tokens == 20
        assert result.total_tokens == 100


# ===========================================================================
# OpenAICompatibleProvider tests
# ===========================================================================


class TestOpenAICompatibleProvider:
    """Unit tests for :class:`OpenAICompatibleProvider`."""

    def _make_provider(self, model_name: str = "test-model") -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model_name=model_name,
        )

    # --- parse_response ---

    def test_openai_parse_response_plain(
        self, openai_raw_response: dict[str, Any]
    ) -> None:
        """Plain response: thinking_content is None, final_answer is correct."""
        provider = self._make_provider()
        result = provider.parse_response(openai_raw_response)

        assert result.thinking_content is None
        assert result.final_answer == "a cat sitting"

    def test_openai_parse_response_with_reasoning_content(
        self, openai_raw_response_with_reasoning: dict[str, Any]
    ) -> None:
        """reasoning_content field is mapped to thinking_content."""
        provider = self._make_provider()
        result = provider.parse_response(openai_raw_response_with_reasoning)

        assert result.thinking_content == "Let me think"
        assert result.final_answer == "a cat sitting"

    def test_openai_parse_response_think_tags(self) -> None:
        """<think> tags embedded in content are parsed into thinking_content."""
        provider = self._make_provider()
        raw: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>reasoning here</think>final answer",
                    }
                }
            ],
            "usage": {"completion_tokens": 10, "total_tokens": 90},
        }
        result = provider.parse_response(raw)

        assert result.thinking_content == "reasoning here"
        assert result.final_answer == "final answer"

    # --- validate_config ---

    def test_openai_validate_config_empty_model_raises(self) -> None:
        """Empty model_name raises ValueError."""
        provider = OpenAICompatibleProvider(model_name="")
        with pytest.raises(ValueError, match="model_name"):
            provider.validate_config()

    # --- build_request ---

    def test_openai_build_request_with_image(self) -> None:
        """When image_data is provided, the content uses the vision array format."""
        provider = self._make_provider()
        payload = provider.build_request("what is this?", image_data="imgbase64==")

        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1]["type"] == "image_url"
        assert "imgbase64==" in content[1]["image_url"]["url"]


# ===========================================================================
# GeminiProvider tests
# ===========================================================================


class TestGeminiProvider:
    """Unit tests for :class:`GeminiProvider`."""

    def _make_provider(
        self, model_name: str = "gemini-2.0-flash", api_key: str = "test-key"
    ) -> GeminiProvider:
        return GeminiProvider(
            api_key=SecretStr(api_key),
            model_name=model_name,
        )

    # --- Identity ---

    def test_gemini_get_provider_type(self) -> None:
        """get_provider_type() returns ProviderType.GEMINI."""
        provider = self._make_provider()
        assert provider.get_provider_type() == ProviderType.GEMINI

    # --- validate_config ---

    def test_gemini_validate_config_no_api_key_raises(self) -> None:
        """When api_key is forced to None after construction, validate raises ValueError."""
        provider = self._make_provider()
        provider.api_key = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="api_key"):
            provider.validate_config()

    # --- parse_response ---

    def test_gemini_parse_response_basic(
        self, gemini_raw_response: dict[str, Any]
    ) -> None:
        """candidatesTokenCount and promptTokenCount are read correctly."""
        provider = self._make_provider()
        result = provider.parse_response(gemini_raw_response)

        assert result.final_answer == "a cat sitting"
        assert result.completion_tokens == 20
        # total = promptTokenCount(80) + candidatesTokenCount(20) = 100
        assert result.total_tokens == 100

    # --- effort → thinking budget ---

    def test_gemini_effort_thinking_budget_mapping(self) -> None:
        """low=1000, medium=5000, high=10000 thinking budget values."""
        assert GeminiProvider._effort_to_thinking_budget("low") == 1000
        assert GeminiProvider._effort_to_thinking_budget("medium") == 5000
        assert GeminiProvider._effort_to_thinking_budget("high") == 10000

    # --- URL construction ---

    def test_gemini_get_api_url(self) -> None:
        """URL contains the model name and ?key= parameter."""
        provider = self._make_provider(
            model_name="gemini-2.0-flash-thinking-exp",
            api_key="my-secret-key",
        )
        url = provider._get_api_url()

        assert "gemini-2.0-flash-thinking-exp" in url
        assert "?key=" in url
        assert "my-secret-key" in url


# ===========================================================================
# SecretStr tests
# ===========================================================================


class TestSecretStr:
    """Unit tests for :class:`SecretStr`."""

    def test_secret_str_repr_masked(self) -> None:
        """repr() returns a masked representation — not the plaintext."""
        secret = SecretStr("sk-supersecret")
        r = repr(secret)
        assert "sk-supersecret" not in r
        assert "**********" in r

    def test_secret_str_str_masked(self) -> None:
        """str() returns the masked string, not the plaintext."""
        secret = SecretStr("sk-supersecret")
        s = str(secret)
        assert s == "**********"
        assert "sk-supersecret" not in s

    def test_secret_str_get_secret_value(self) -> None:
        """get_secret_value() returns the plaintext."""
        secret = SecretStr("sk-supersecret")
        assert secret.get_secret_value() == "sk-supersecret"
