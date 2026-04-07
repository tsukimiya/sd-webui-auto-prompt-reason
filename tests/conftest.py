"""
Shared pytest fixtures for sd-webui-auto-prompt-reason tests.

Provides:
- ``ReasoningResponse`` sample instances (with and without thinking content)
- Raw API response dicts for all three providers:
    - Ollama Shape A (≥ 0.7.0, dedicated ``thinking`` field)
    - Ollama Shape B (< 0.7.0, ``<think>`` tags in ``content``)
    - OpenAI-compatible (plain and with ``reasoning_content``)
    - Gemini ``generateContent``
- Mock provider objects for ``OllamaProvider``, ``OpenAICompatibleProvider``,
  and ``GeminiProvider`` that return a canned ``ReasoningResponse``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest  # type: ignore[import-untyped]

from apr_modules.models.reasoning_response import ReasoningResponse  # type: ignore[import]
from apr_modules.providers.ollama_provider import OllamaProvider  # type: ignore[import]
from apr_modules.providers.openai_provider import OpenAICompatibleProvider  # type: ignore[import]
from apr_modules.providers.gemini_provider import GeminiProvider  # type: ignore[import]


# ---------------------------------------------------------------------------
# ReasoningResponse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_reasoning_response() -> ReasoningResponse:
    """A fully-populated ``ReasoningResponse`` with thinking content."""
    return ReasoningResponse(
        final_answer="a cat sitting on a bench",
        thinking_content="Let me think about this carefully",
        reasoning_tokens=5,
        completion_tokens=20,
        total_tokens=100,
        reasoning_time_ms=1500,
        model="test-model",
        provider="ollama",
        raw_response={},
    )


@pytest.fixture
def sample_reasoning_response_no_thinking() -> ReasoningResponse:
    """A ``ReasoningResponse`` with no thinking content."""
    return ReasoningResponse(
        final_answer="a cat sitting on a bench",
        thinking_content=None,
        reasoning_tokens=None,
        completion_tokens=20,
        total_tokens=100,
        reasoning_time_ms=1500,
        model="test-model",
        provider="ollama",
        raw_response={},
    )


# ---------------------------------------------------------------------------
# Raw API response fixtures — Ollama
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_raw_shape_a() -> dict[str, Any]:
    """Ollama ≥ 0.7.0 response: dedicated ``thinking`` field in ``message``."""
    return {
        "message": {
            "role": "assistant",
            "content": "a cat sitting",
            "thinking": "Let me think",
        },
        "eval_count": 20,
        "prompt_eval_count": 80,
    }


@pytest.fixture
def ollama_raw_shape_b() -> dict[str, Any]:
    """Ollama < 0.7.0 response: ``<think>`` tags embedded in ``content``."""
    return {
        "message": {
            "role": "assistant",
            "content": "<think>Let me think</think>a cat sitting",
        },
        "eval_count": 20,
        "prompt_eval_count": 80,
    }


# ---------------------------------------------------------------------------
# Raw API response fixtures — Ollama via /v1/chat/completions (OpenAI-compat)
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_v1_shape_a() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape A: dedicated ``thinking`` field."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "qwen3.5:7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a cat sitting",
                    "thinking": "Let me think",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


@pytest.fixture
def ollama_v1_shape_b() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape B: think tags in content."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "qwen3.5:7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think Let me think</think a cat sitting",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


@pytest.fixture
def ollama_v1_shape_c() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape C: ``reasoning`` field.

    This is the format Ollama uses for thinking models (e.g. qwen3.5)
    when accessed via the OpenAI-compatible endpoint.
    ``content`` may be empty when all tokens are consumed by reasoning.
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "jaahas/qwen3.5-uncensored",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a cat sitting on a bench",
                    "reasoning": "Let me think about this prompt carefully",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


@pytest.fixture
def ollama_v1_shape_c_empty_content() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape C: reasoning field with empty content.

    Known Ollama bug: content is empty when thinking model consumes all tokens.
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "jaahas/qwen3.5-uncensored",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "The user wants a prompt for image generation...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 3508,
            "total_tokens": 3588,
        },
    }


@pytest.fixture
def ollama_v1_shape_c_null_content() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape C: content is null/None."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "jaahas/qwen3.5-uncensored",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Analyzing the image prompt requirements...",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 100,
            "total_tokens": 180,
        },
    }


@pytest.fixture
def ollama_v1_shape_d() -> dict[str, Any]:
    """Ollama via /v1/chat/completions Shape D: ``reasoning_content`` field (DeepSeek-style)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a cat sitting",
                    "reasoning_content": "Deep reasoning trace here",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


# ---------------------------------------------------------------------------
# Raw API response fixtures — OpenAI-compatible
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_raw_response() -> dict[str, Any]:
    """Standard OpenAI ``/v1/chat/completions`` response without reasoning."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a cat sitting",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


@pytest.fixture
def openai_raw_response_with_reasoning() -> dict[str, Any]:
    """OpenAI-compatible response that also carries ``reasoning_content`` (e.g. DeepSeek R1)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a cat sitting",
                    "reasoning_content": "Let me think",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
        },
    }


# ---------------------------------------------------------------------------
# Raw API response fixtures — Gemini
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_raw_response() -> dict[str, Any]:
    """Gemini ``generateContent`` response."""
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "a cat sitting"},
                    ],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 80,
            "candidatesTokenCount": 20,
            "totalTokenCount": 100,
        },
    }


# ---------------------------------------------------------------------------
# Mock provider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_provider(sample_reasoning_response: ReasoningResponse) -> MagicMock:
    """A ``MagicMock`` mimicking ``OllamaProvider``.

    ``parse_response()`` returns ``sample_reasoning_response``.
    """
    mock = MagicMock(spec=OllamaProvider)
    mock.parse_response.return_value = sample_reasoning_response
    return mock


@pytest.fixture
def mock_openai_provider(sample_reasoning_response: ReasoningResponse) -> MagicMock:
    """A ``MagicMock`` mimicking ``OpenAICompatibleProvider``.

    ``parse_response()`` returns ``sample_reasoning_response``.
    """
    mock = MagicMock(spec=OpenAICompatibleProvider)
    mock.parse_response.return_value = sample_reasoning_response
    return mock


@pytest.fixture
def mock_gemini_provider(sample_reasoning_response: ReasoningResponse) -> MagicMock:
    """A ``MagicMock`` mimicking ``GeminiProvider``.

    ``parse_response()`` returns ``sample_reasoning_response``.
    """
    mock = MagicMock(spec=GeminiProvider)
    mock.parse_response.return_value = sample_reasoning_response
    return mock
