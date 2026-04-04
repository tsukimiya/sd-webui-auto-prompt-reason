"""
Reasoning parser and normalizer for sd-webui-auto-prompt-reason.

This module provides the `ReasoningParser` class, which normalizes raw API
responses from all supported providers (Ollama, OpenAI-compatible, Gemini)
into a unified `ReasoningResponse` object.  No HTTP calls are made here —
this is pure parsing / data-transformation logic.
"""

from typing import Optional

from apr_modules.models.reasoning_response import ReasoningResponse


class ReasoningParser:
    """Parse and normalize raw LLM API responses from supported providers.

    All methods are static; the class acts as a namespace for related parsing
    helpers rather than holding any mutable state.
    """

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_think_tags(content: str) -> tuple[Optional[str], str]:
        """Extract ``<think>...</think>`` thinking content from a string.

        Uses plain string methods — no regex.

        Parameters
        ----------
        content:
            Raw text that may contain ``<think>`` tags.

        Returns
        -------
        tuple[Optional[str], str]
            ``(thinking, remaining_text)`` where *thinking* is the text inside
            the tags (or the text after a lone opening tag), and
            *remaining_text* is everything after ``</think>`` (stripped).

            * Both tags present  → ``(thinking, remaining.strip())``
            * Only ``<think>``   → ``(content[think_start+7:], "")``
            * Neither tag        → ``(None, content)``
        """
        # <think>  = 7 chars
        # </think> = 8 chars
        think_start = content.find("<think>")
        think_end = content.find("</think>")

        if think_start != -1 and think_end != -1:
            thinking = content[think_start + 7 : think_end]
            final = content[think_end + 8 :].strip()
            return thinking, final
        elif think_start != -1:
            # Truncated response — no closing tag
            thinking = content[think_start + 7 :]
            return thinking, ""
        else:
            return None, content

    # ------------------------------------------------------------------
    # Provider-specific normalizers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ollama(raw: dict) -> dict:
        """Normalize a raw Ollama API response dict.

        Supports two shapes:

        * **≥ 0.7.0** — ``raw["message"]["thinking"]`` exists: thinking is
          taken directly from that field and content from
          ``raw["message"]["content"]``.
        * **< 0.7.0** — ``raw["message"]["content"]`` contains embedded
          ``<think>`` tags which are extracted via
          :meth:`parse_think_tags`.

        Parameters
        ----------
        raw:
            The JSON-decoded response body from an Ollama ``/api/chat`` call.

        Returns
        -------
        dict
            Keys: ``thinking`` (``Optional[str]``), ``final_answer``
            (``str``), ``completion_tokens`` (``int``),
            ``prompt_tokens`` (``int``).
        """
        message = raw.get("message", {})

        if "thinking" in message:
            # Ollama ≥ 0.7.0 — dedicated thinking field
            thinking: Optional[str] = message.get("thinking") or None
            final_answer: str = message.get("content", "")
        else:
            # Ollama < 0.7.0 — <think> tags embedded in content
            thinking, final_answer = ReasoningParser.parse_think_tags(
                message.get("content", "")
            )

        return {
            "thinking": thinking,
            "final_answer": final_answer,
            "completion_tokens": int(raw.get("eval_count", 0)),
            "prompt_tokens": int(raw.get("prompt_eval_count", 0)),
        }

    @staticmethod
    def normalize_openai(raw: dict) -> dict:
        """Normalize a raw OpenAI-compatible API response dict.

        Checks for an explicit ``reasoning_content`` field first (e.g.,
        DeepSeek-R1 via OpenAI-compatible proxy), then falls back to
        extracting ``<think>`` tags from the message content.

        Parameters
        ----------
        raw:
            The JSON-decoded response body from an OpenAI ``/v1/chat/completions``
            compatible endpoint.

        Returns
        -------
        dict
            Keys: ``thinking`` (``Optional[str]``), ``final_answer``
            (``str``), ``completion_tokens`` (``int``),
            ``total_tokens`` (``int``).
        """
        choices = raw.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}

        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            thinking = reasoning_content
            final_answer = message.get("content", "")
        else:
            thinking, final_answer = ReasoningParser.parse_think_tags(
                message.get("content", "")
            )

        usage = raw.get("usage", {})
        return {
            "thinking": thinking,
            "final_answer": final_answer,
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }

    @staticmethod
    def normalize_gemini(raw: dict) -> dict:
        """Normalize a raw Google Gemini API response dict.

        Concatenates all ``text`` values from
        ``candidates[0].content.parts[]``, then extracts ``<think>`` tags
        from the combined text.

        Parameters
        ----------
        raw:
            The JSON-decoded response body from a Gemini
            ``generateContent`` call.

        Returns
        -------
        dict
            Keys: ``thinking`` (``Optional[str]``), ``final_answer``
            (``str``), ``completion_tokens`` (``int``),
            ``reasoning_tokens`` (``Optional[int]``),
            ``total_tokens`` (``int``).
        """
        candidates = raw.get("candidates", [])
        parts: list = []
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

        full_text = "".join(part.get("text", "") for part in parts)
        thinking, final_answer = ReasoningParser.parse_think_tags(full_text)

        usage = raw.get("usageMetadata", {})
        completion_tokens = int(usage.get("candidatesTokenCount", 0))
        raw_thoughts = usage.get("thoughtsTokenCount")
        reasoning_tokens: Optional[int] = int(raw_thoughts) if raw_thoughts is not None else None
        total_tokens = int(usage.get("promptTokenCount", 0)) + completion_tokens

        return {
            "thinking": thinking,
            "final_answer": final_answer,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_thinking_ratio(
        thinking_content: Optional[str], final_answer: str
    ) -> float:
        """Return the fraction of total characters that are thinking content.

        Parameters
        ----------
        thinking_content:
            The extracted thinking text, or ``None`` if not available.
        final_answer:
            The model's final answer text.

        Returns
        -------
        float
            A value in ``[0.0, 1.0]``.  Returns ``0.0`` when
            *thinking_content* is ``None`` or the combined length is zero.
        """
        if not thinking_content:
            return 0.0
        total = len(thinking_content) + len(final_answer)
        if total == 0:
            return 0.0
        return len(thinking_content) / total

    @staticmethod
    def build_response(
        provider: str,
        model: str,
        thinking: Optional[str],
        final_answer: str,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: Optional[int],
        reasoning_time_ms: int,
    ) -> ReasoningResponse:
        """Construct a :class:`~modules.models.reasoning_response.ReasoningResponse`.

        Parameters
        ----------
        provider:
            Provider identifier string, e.g. ``"ollama"``, ``"openai"``,
            ``"gemini"``.
        model:
            Model name/identifier as returned by or configured for the
            provider.
        thinking:
            Extracted thinking/reasoning text, or ``None``.
        final_answer:
            The model's final answer (without ``<think>`` tags).
        completion_tokens:
            Number of tokens in the completion.
        total_tokens:
            Total tokens consumed (prompt + completion).
        reasoning_tokens:
            Tokens attributed to the thinking step, or ``None``.
        reasoning_time_ms:
            Wall-clock time spent waiting for the API response, in
            milliseconds.

        Returns
        -------
        ReasoningResponse
            A fully populated response object ready for downstream use.
        """
        return ReasoningResponse(
            final_answer=final_answer,
            thinking_content=thinking,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_time_ms=reasoning_time_ms,
            model=model,
            provider=provider,
            raw_response={},
        )
