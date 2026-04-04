"""
Comprehensive unit tests for ``ReasoningParser``.

Tests cover all 6 static methods:
- ``parse_think_tags``       (5 cases)
- ``normalize_ollama``       (5 cases)
- ``normalize_openai``       (5 cases)
- ``normalize_gemini``       (5 cases)
- ``calculate_thinking_ratio`` (5 cases)
- ``build_response``          (4 cases)

Total: 29 test cases, zero live HTTP calls.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest  # type: ignore[import-untyped]

from modules.reasoning_parser import ReasoningParser  # type: ignore[import]
from modules.models.reasoning_response import ReasoningResponse  # type: ignore[import]


# ===========================================================================
# TestParseThinkTags
# ===========================================================================


class TestParseThinkTags:
    """Unit tests for :meth:`ReasoningParser.parse_think_tags`."""

    def test_both_tags_present(self) -> None:
        """Both ``<think>`` and ``</think>`` present: thinking is extracted, remaining is stripped."""
        thinking, remaining = ReasoningParser.parse_think_tags("<think>ideas</think>answer")
        assert thinking == "ideas"
        assert remaining == "answer"

    def test_both_tags_with_whitespace_in_remaining(self) -> None:
        """Remaining text after ``</think>`` is stripped of leading/trailing whitespace."""
        thinking, remaining = ReasoningParser.parse_think_tags("<think>step1</think>  final text  ")
        assert thinking == "step1"
        assert remaining == "final text"

    def test_only_open_tag_truncated(self) -> None:
        """Only ``<think>`` present (truncated response): returns thinking with empty remaining."""
        thinking, remaining = ReasoningParser.parse_think_tags("<think>partial thought")
        assert thinking == "partial thought"
        assert remaining == ""

    def test_no_tags_at_all(self) -> None:
        """No ``<think>`` tags at all: returns ``(None, original_content)``."""
        thinking, remaining = ReasoningParser.parse_think_tags("plain content")
        assert thinking is None
        assert remaining == "plain content"

    def test_empty_string(self) -> None:
        """Empty string input returns ``(None, "")``."""
        thinking, remaining = ReasoningParser.parse_think_tags("")
        assert thinking is None
        assert remaining == ""

    def test_think_tag_with_empty_content(self) -> None:
        """``<think></think>`` yields empty-string thinking and stripped remaining."""
        thinking, remaining = ReasoningParser.parse_think_tags("<think></think>answer here")
        assert thinking == ""
        assert remaining == "answer here"

    def test_multiline_thinking(self) -> None:
        """Multi-line thinking content is returned verbatim."""
        content = "<think>line one\nline two\nline three</think>done"
        thinking, remaining = ReasoningParser.parse_think_tags(content)
        assert thinking == "line one\nline two\nline three"
        assert remaining == "done"


# ===========================================================================
# TestNormalizeOllama
# ===========================================================================


class TestNormalizeOllama:
    """Unit tests for :meth:`ReasoningParser.normalize_ollama`."""

    def test_shape_a_thinking_field(self, ollama_raw_shape_a: dict[str, Any]) -> None:
        """Shape A (≥0.7.0): ``thinking`` field is read directly from ``message``."""
        result = ReasoningParser.normalize_ollama(ollama_raw_shape_a)
        assert result["thinking"] == "Let me think"
        assert result["final_answer"] == "a cat sitting"

    def test_shape_a_token_counts(self, ollama_raw_shape_a: dict[str, Any]) -> None:
        """Shape A: ``eval_count`` and ``prompt_eval_count`` map to token fields."""
        result = ReasoningParser.normalize_ollama(ollama_raw_shape_a)
        assert result["completion_tokens"] == 20
        assert result["prompt_tokens"] == 80

    def test_shape_b_think_tags(self, ollama_raw_shape_b: dict[str, Any]) -> None:
        """Shape B (<0.7.0): ``<think>`` tags in ``content`` are extracted."""
        result = ReasoningParser.normalize_ollama(ollama_raw_shape_b)
        assert result["thinking"] == "Let me think"
        assert result["final_answer"] == "a cat sitting"

    def test_shape_b_no_tags_returns_none_thinking(self) -> None:
        """Shape B with plain content (no tags): thinking is None, final_answer is the content."""
        raw: dict[str, Any] = {
            "message": {"role": "assistant", "content": "just a plain answer"},
            "eval_count": 5,
            "prompt_eval_count": 15,
        }
        result = ReasoningParser.normalize_ollama(raw)
        assert result["thinking"] is None
        assert result["final_answer"] == "just a plain answer"

    def test_missing_message_key(self) -> None:
        """Missing ``message`` key in raw response: returns defaults without raising."""
        raw: dict[str, Any] = {"eval_count": 3, "prompt_eval_count": 7}
        result = ReasoningParser.normalize_ollama(raw)
        assert result["thinking"] is None
        assert result["final_answer"] == ""
        assert result["completion_tokens"] == 3
        assert result["prompt_tokens"] == 7

    def test_shape_a_empty_thinking_field_becomes_none(self) -> None:
        """Shape A: empty string ``thinking`` field is coerced to ``None``."""
        raw: dict[str, Any] = {
            "message": {"role": "assistant", "content": "answer", "thinking": ""},
            "eval_count": 10,
            "prompt_eval_count": 50,
        }
        result = ReasoningParser.normalize_ollama(raw)
        assert result["thinking"] is None
        assert result["final_answer"] == "answer"


# ===========================================================================
# TestNormalizeOpenAI
# ===========================================================================


class TestNormalizeOpenAI:
    """Unit tests for :meth:`ReasoningParser.normalize_openai`."""

    def test_with_reasoning_content(
        self, openai_raw_response_with_reasoning: dict[str, Any]
    ) -> None:
        """``reasoning_content`` field takes priority over ``<think>`` tag parsing."""
        result = ReasoningParser.normalize_openai(openai_raw_response_with_reasoning)
        assert result["thinking"] == "Let me think"
        assert result["final_answer"] == "a cat sitting"

    def test_plain_response_no_thinking(self, openai_raw_response: dict[str, Any]) -> None:
        """Standard response without reasoning: thinking is ``None``."""
        result = ReasoningParser.normalize_openai(openai_raw_response)
        assert result["thinking"] is None
        assert result["final_answer"] == "a cat sitting"

    def test_think_tags_in_content(self) -> None:
        """When no ``reasoning_content``, ``<think>`` tags in content are extracted."""
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
        result = ReasoningParser.normalize_openai(raw)
        assert result["thinking"] == "reasoning here"
        assert result["final_answer"] == "final answer"

    def test_token_counts(self, openai_raw_response_with_reasoning: dict[str, Any]) -> None:
        """``completion_tokens`` and ``total_tokens`` are read from ``usage``."""
        result = ReasoningParser.normalize_openai(openai_raw_response_with_reasoning)
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 100

    def test_empty_choices_returns_none_thinking(self) -> None:
        """When ``choices`` is empty, thinking is None and final_answer is empty string."""
        raw: dict[str, Any] = {
            "choices": [],
            "usage": {"completion_tokens": 0, "total_tokens": 0},
        }
        result = ReasoningParser.normalize_openai(raw)
        assert result["thinking"] is None
        assert result["final_answer"] == ""


# ===========================================================================
# TestNormalizeGemini
# ===========================================================================


class TestNormalizeGemini:
    """Unit tests for :meth:`ReasoningParser.normalize_gemini`."""

    def test_basic_response(self, gemini_raw_response: dict[str, Any]) -> None:
        """Basic Gemini response: final_answer and token counts parsed correctly."""
        result = ReasoningParser.normalize_gemini(gemini_raw_response)
        assert result["final_answer"] == "a cat sitting"
        assert result["thinking"] is None

    def test_token_counts(self, gemini_raw_response: dict[str, Any]) -> None:
        """``candidatesTokenCount`` and ``promptTokenCount`` map to token fields."""
        result = ReasoningParser.normalize_gemini(gemini_raw_response)
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 100  # 80 + 20

    def test_with_thoughts_token_count(self) -> None:
        """``thoughtsTokenCount`` in ``usageMetadata`` is stored as ``reasoning_tokens``."""
        raw: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "<think>deep thoughts</think>the answer"}],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 30,
                "thoughtsTokenCount": 12,
            },
        }
        result = ReasoningParser.normalize_gemini(raw)
        assert result["reasoning_tokens"] == 12
        assert result["thinking"] == "deep thoughts"
        assert result["final_answer"] == "the answer"

    def test_without_thoughts_token_count(self, gemini_raw_response: dict[str, Any]) -> None:
        """When ``thoughtsTokenCount`` is absent, ``reasoning_tokens`` is ``None``."""
        result = ReasoningParser.normalize_gemini(gemini_raw_response)
        assert result["reasoning_tokens"] is None

    def test_empty_candidates(self) -> None:
        """Empty ``candidates`` list: thinking is None, final_answer is empty string."""
        raw: dict[str, Any] = {
            "candidates": [],
            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0},
        }
        result = ReasoningParser.normalize_gemini(raw)
        assert result["thinking"] is None
        assert result["final_answer"] == ""
        assert result["completion_tokens"] == 0

    def test_multiple_parts_concatenated(self) -> None:
        """Multiple ``parts`` entries are joined before tag extraction."""
        raw: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "<think>"},
                            {"text": "combined thinking"},
                            {"text": "</think>combined answer"},
                        ],
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10},
        }
        result = ReasoningParser.normalize_gemini(raw)
        assert result["thinking"] == "combined thinking"
        assert result["final_answer"] == "combined answer"


# ===========================================================================
# TestCalculateThinkingRatio
# ===========================================================================


class TestCalculateThinkingRatio:
    """Unit tests for :meth:`ReasoningParser.calculate_thinking_ratio`."""

    def test_normal_ratio(self) -> None:
        """Returns correct fraction when both thinking and final_answer are non-empty."""
        # thinking=4 chars, final_answer=6 chars → 4/10 = 0.4
        ratio = ReasoningParser.calculate_thinking_ratio("abcd", "efghij")
        assert ratio == pytest.approx(0.4)

    def test_thinking_is_none(self) -> None:
        """When ``thinking_content`` is ``None``, ratio is 0.0."""
        ratio = ReasoningParser.calculate_thinking_ratio(None, "some answer")
        assert ratio == 0.0

    def test_thinking_is_empty_string(self) -> None:
        """When ``thinking_content`` is an empty string, ratio is 0.0."""
        ratio = ReasoningParser.calculate_thinking_ratio("", "some answer")
        assert ratio == 0.0

    def test_both_empty_strings(self) -> None:
        """When both arguments are empty, ratio is 0.0 (no division by zero)."""
        ratio = ReasoningParser.calculate_thinking_ratio("", "")
        assert ratio == 0.0

    def test_full_thinking_no_final(self) -> None:
        """When final_answer is empty and thinking is non-empty, ratio is 1.0."""
        ratio = ReasoningParser.calculate_thinking_ratio("some thinking", "")
        assert ratio == pytest.approx(1.0)

    def test_ratio_in_bounds(self) -> None:
        """Ratio is always in [0.0, 1.0] for realistic inputs."""
        ratio = ReasoningParser.calculate_thinking_ratio("short", "a much longer answer here")
        assert 0.0 <= ratio <= 1.0


# ===========================================================================
# TestBuildResponse
# ===========================================================================


class TestBuildResponse:
    """Unit tests for :meth:`ReasoningParser.build_response`."""

    def _build(
        self,
        provider: str = "ollama",
        model: str = "test-model",
        thinking: Optional[str] = "Let me think",
        final_answer: str = "a cat sitting",
        completion_tokens: int = 20,
        total_tokens: int = 100,
        reasoning_tokens: Optional[int] = 5,
        reasoning_time_ms: int = 1500,
    ) -> ReasoningResponse:
        return ReasoningParser.build_response(
            provider=provider,
            model=model,
            thinking=thinking,
            final_answer=final_answer,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_time_ms=reasoning_time_ms,
        )

    def test_returns_reasoning_response_type(self) -> None:
        """``build_response()`` returns a ``ReasoningResponse`` instance."""
        result = self._build()
        assert isinstance(result, ReasoningResponse)

    def test_all_fields_populated_correctly(self) -> None:
        """All fields of the returned ``ReasoningResponse`` match the inputs."""
        result = self._build(
            provider="gemini",
            model="gemini-2.0-flash",
            thinking="deep reasoning",
            final_answer="final output",
            completion_tokens=30,
            total_tokens=120,
            reasoning_tokens=8,
            reasoning_time_ms=2000,
        )
        assert result.provider == "gemini"
        assert result.model == "gemini-2.0-flash"
        assert result.thinking_content == "deep reasoning"
        assert result.final_answer == "final output"
        assert result.completion_tokens == 30
        assert result.total_tokens == 120
        assert result.reasoning_tokens == 8
        assert result.reasoning_time_ms == 2000

    def test_thinking_none_propagates(self) -> None:
        """``thinking=None`` is stored as ``thinking_content=None`` in the response."""
        result = self._build(thinking=None, reasoning_tokens=None)
        assert result.thinking_content is None
        assert result.reasoning_tokens is None
        assert result.has_thinking is False

    def test_raw_response_is_empty_dict(self) -> None:
        """``raw_response`` is always set to ``{}`` by ``build_response``."""
        result = self._build()
        assert result.raw_response == {}
