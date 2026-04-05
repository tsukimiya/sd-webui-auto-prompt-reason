"""
Ollama provider implementation for sd-webui-auto-prompt-reason.

This module implements the :class:`OllamaProvider` concrete class that talks to
a locally-running Ollama inference server (default: ``http://localhost:11434``).

Response parsing
----------------
Four Ollama response shapes are supported:

* **Shape A** (native ``/api/chat``, Ollama ≥ 0.7.0): the ``message`` object
  contains a dedicated ``thinking`` field alongside ``content``.
* **Shape B** (older Ollama): chain-of-thought is embedded in ``content``
  between ``<think …>`` / ``</think >`` tags.
* **Shape C** (``/v1/chat/completions`` with thinking model): Ollama exposes
  a ``reasoning`` field in the message object.  ``content`` may be empty when
  all tokens are consumed by reasoning.
* **Shape D** (DeepSeek-style): ``reasoning_content`` field in the message.

Security note
-------------
``api_key`` is stored as :class:`~modules.providers.base_provider.SecretStr`
and is **never** logged or printed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import BaseProvider, ProviderType, SecretStr

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"

# Effort → temperature mapping (lower temperature = more focused / deterministic)
_EFFORT_TEMPERATURE: dict[str, float] = {
    "low": 0.8,
    "medium": 0.5,
    "high": 0.2,
}


class OllamaProvider(BaseProvider):
    """Concrete :class:`BaseProvider` for the Ollama local inference server.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama API. Defaults to ``"http://localhost:11434"``.
        Pass an empty string to use the default.
    model_name:
        Ollama model tag, e.g. ``"qwen3.5:7b"`` or ``"llama3.2:latest"``.
    api_key:
        Optional bearer token.  Most local Ollama installations do not
        require authentication; pass ``None`` (the default).
    timeout:
        ``(connect_timeout, read_timeout)`` in seconds.  Defaults to
        ``(10, 180)``.

    Examples
    --------
    >>> provider = OllamaProvider(
    ...     base_url="http://localhost:11434",
    ...     model_name="qwen3.5:7b",
    ... )
    >>> provider.get_provider_type()
    <ProviderType.OLLAMA: 'ollama'>
    >>> provider.supports_reasoning()
    True
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model_name: str = "",
        api_key: Optional[SecretStr] = None,
        timeout: tuple = (10, 180),
    ) -> None:
        effective_url = base_url if base_url else _DEFAULT_BASE_URL
        super().__init__(
            base_url=effective_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Raise :py:exc:`ValueError` if ``model_name`` is empty.

        Raises
        ------
        ValueError
            If :attr:`model_name` is an empty string.
        """
        if not self.model_name:
            raise ValueError("model_name cannot be empty")

    def get_provider_type(self) -> ProviderType:
        """Return :attr:`ProviderType.OLLAMA`."""
        return ProviderType.OLLAMA

    def supports_reasoning(self) -> bool:
        """Return ``True`` — Ollama models expose reasoning content."""
        return True

    def build_request(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Build an Ollama ``/api/chat`` request payload.

        Parameters
        ----------
        prompt:
            User-facing text prompt.
        image_data:
            Optional base-64-encoded image bytes.  When provided, the image is
            attached to the message using the Ollama ``images`` field.
        reasoning_effort:
            One of ``"low"``, ``"medium"``, or ``"high"``.  Controls the
            sampling temperature sent to the model.
        system_prompt:
            Optional system-level instruction.  When non-empty, a
            ``{"role": "system", …}`` message is prepended to ``messages``.

        Returns
        -------
        dict
            JSON-serialisable payload ready to POST to ``/api/chat``.
        """
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_data is not None:
            message["images"] = [image_data]

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(message)

        return {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": self._effort_to_temperature(reasoning_effort),
            # max_tokens は意図的に省略 — thinking モデルは reasoning だけで
            # トークン上限を使い切り content が空になる問題を防ぐため、
            # Ollama のモデル別デフォルト上限に委ねる。
        }

    def parse_response(self, raw_response: dict) -> ReasoningResponse:
        """Parse an Ollama API response into a :class:`ReasoningResponse`.

        Handles four response shapes:

        * **Shape A** — Native ``/api/chat`` (Ollama ≥ 0.7.0):
          ``message.thinking`` contains the reasoning trace.
        * **Shape B** — ``<think …>`` tags embedded in ``message.content``
          (older Ollama versions).
        * **Shape C** — ``/v1/chat/completions`` with thinking model:
          ``message.reasoning`` contains the reasoning trace; ``message.content``
          may be empty when all tokens are consumed by reasoning.
        * **Shape D** — ``reasoning_content`` field (DeepSeek-style providers).

        Parameters
        ----------
        raw_response:
            Deserialised JSON body from the Ollama ``/v1/chat/completions`` endpoint.

        Returns
        -------
        ReasoningResponse
            Normalised provider-agnostic response.
        """
        # --- 診断ログ: 生レスポンス構造の確認 ---
        # OpenAI互換形式: choices[0].message.content
        choices: list = raw_response.get("choices", [])
        message: dict[str, Any] = choices[0].get("message", {}) if choices else {}
        # content may be None when thinking model puts all text in reasoning field
        raw_content: Any = message.get("content")
        content: str = raw_content if isinstance(raw_content, str) else (raw_content or "")

        logger.warning(
            "[apr][ollama][parse] raw_keys=%s choices_len=%d message_keys=%s"
            " content_len=%s content_preview=%.200r"
            " has_thinking=%s has_reasoning=%s has_reasoning_content=%s",
            sorted(raw_response.keys()) if isinstance(raw_response, dict) else type(raw_response).__name__,
            len(choices),
            sorted(message.keys()) if isinstance(message, dict) else type(message).__name__,
            len(content) if isinstance(raw_content, str) else repr(type(raw_content).__name__),
            content[:200],
            "thinking" in message,
            "reasoning" in message,
            "reasoning_content" in message,
        )

        if not choices:
            logger.warning(
                "[apr][ollama][parse] 'choices' field missing or empty — "
                "raw_keys=%s  (wrong endpoint or unexpected response format?)",
                sorted(raw_response.keys()) if isinstance(raw_response, dict) else type(raw_response).__name__,
            )

        thinking_content: Optional[str] = None
        final_answer: str = content
        _shape_detected: str = "plain(no thinking)"

        # --- Shape A: Native /api/chat — dedicated ``thinking`` field ---
        if "thinking" in message:
            _shape_detected = "A(dedicated thinking field)"
            thinking_content = message["thinking"] or None
            final_answer = content
            logger.warning(
                "[apr][ollama][parse] Shape-A: thinking_len=%s thinking_preview=%.200r"
                " content_len=%d content_preview=%.200r",
                len(thinking_content) if thinking_content else "None",
                (thinking_content or "")[:200],
                len(content),
                content[:200],
            )

        # --- Shape C: /v1/chat/completions — ``reasoning`` field ---
        # Ollama uses ``reasoning`` for thinking content in the OpenAI-compatible
        # endpoint.  ``content`` may be empty when the model spends all tokens
        # on reasoning.
        elif "reasoning" in message:
            _shape_detected = "C(reasoning field - OpenAI compat)"
            reasoning_raw: Any = message.get("reasoning")
            thinking_content = reasoning_raw if isinstance(reasoning_raw, str) and reasoning_raw else None
            final_answer = content
            logger.warning(
                "[apr][ollama][parse] Shape-C: reasoning_len=%s reasoning_preview=%.200r"
                " content_len=%d content_preview=%.200r",
                len(reasoning_raw) if isinstance(reasoning_raw, str) else repr(type(reasoning_raw).__name__),
                (reasoning_raw or "")[:200] if isinstance(reasoning_raw, str) else reasoning_raw,
                len(content),
                content[:200],
            )
            if not content.strip():
                logger.warning(
                    "[apr][ollama][parse] Shape-C: content is EMPTY — "
                    "final_answer will be empty. reasoning_len=%s",
                    len(reasoning_raw) if isinstance(reasoning_raw, str) else repr(type(reasoning_raw).__name__),
                )

        # --- Shape D: ``reasoning_content`` field (DeepSeek-style) ---
        elif "reasoning_content" in message:
            _shape_detected = "D(reasoning_content field)"
            rc_raw: Any = message.get("reasoning_content")
            thinking_content = rc_raw if isinstance(rc_raw, str) and rc_raw else None
            final_answer = content
            logger.warning(
                "[apr][ollama][parse] Shape-D: reasoning_content_len=%s reasoning_content_preview=%.200r"
                " content_len=%d content_preview=%.200r",
                len(rc_raw) if isinstance(rc_raw, str) else repr(type(rc_raw).__name__),
                (rc_raw or "")[:200] if isinstance(rc_raw, str) else rc_raw,
                len(content),
                content[:200],
            )

        # --- Shape B: thinking embedded in <think…> tags ---
        else:
            think_start = content.find("<think")
            think_end = content.find("</think")

            if think_start != -1 and think_end != -1:
                # Both tags present — clean extraction
                # Find the closing '>' of the opening tag to skip attributes
                tag_close = content.find(">", think_start)
                if tag_close != -1 and tag_close < think_end:
                    _shape_detected = "B(both think tags)"
                    thinking_content = content[tag_close + 1 : think_end]
                    final_answer = content[think_end + 8 :].strip()
                    logger.warning(
                        "[apr][ollama][parse] Shape-B(both tags): thinking_len=%d thinking_preview=%.200r"
                        " final_answer_len=%d final_answer_preview=%.200r",
                        len(thinking_content),
                        thinking_content[:200],
                        len(final_answer),
                        final_answer[:200],
                    )
                else:
                    # Malformed tag — treat as no thinking
                    _shape_detected = "plain(malformed think tag)"
                    logger.warning(
                        "[apr][ollama][parse] Shape-B: malformed think tag — "
                        "treating as plain content. content_preview=%.200r",
                        content[:200],
                    )
            elif think_start != -1:
                # Opening tag only — response was truncated
                tag_close = content.find(">", think_start)
                if tag_close != -1:
                    _shape_detected = "B(unclosed think tag — truncated)"
                    thinking_content = content[tag_close + 1 :]
                    final_answer = ""
                    logger.warning(
                        "[apr][ollama][parse] Shape-B(unclosed/truncated): thinking_len=%d thinking_preview=%.200r",
                        len(thinking_content),
                        thinking_content[:200],
                    )
                else:
                    _shape_detected = "plain(malformed think tag — no closing >)"
                    logger.warning(
                        "[apr][ollama][parse] plain(no closing >): content_preview=%.200r",
                        content[:200],
                    )
            else:
                # No think tags at all — plain response
                logger.warning(
                    "[apr][ollama][parse] plain(no thinking tags): content_len=%d content_preview=%.200r",
                    len(content),
                    content[:200],
                )
            # else: no tags — keep thinking_content=None, final_answer=content

        # OpenAI互換レスポンスのトークンカウント
        usage: dict[str, Any] = raw_response.get("usage", {})
        completion_tokens: int = usage.get("completion_tokens", 0)
        prompt_tokens: int = usage.get("prompt_tokens", 0)

        # --- 診断ログ: パース結果の確認 ---
        logger.warning(
            "[apr][ollama][parse] RESULT: shape=%s final_answer_len=%d"
            " final_answer_preview=%.200r thinking_len=%s"
            " completion_tokens=%d prompt_tokens=%d",
            _shape_detected,
            len(final_answer.strip()),
            final_answer.strip()[:200],
            len(thinking_content) if thinking_content else "None",
            completion_tokens,
            prompt_tokens,
        )

        if not final_answer.strip():
            logger.warning(
                "[apr][ollama][parse] WARNING: final_answer is EMPTY after shape=%s — "
                "prompt injection will have no effect!",
                _shape_detected,
            )

        return ReasoningResponse(
            final_answer=final_answer.strip(),
            thinking_content=thinking_content or None,
            reasoning_tokens=(
                len(thinking_content.split()) if thinking_content else None
            ),
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens + prompt_tokens,
            reasoning_time_ms=0,  # caller is responsible for setting this
            model=self.model_name,
            provider="ollama",
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effort_to_temperature(self, reasoning_effort: str) -> float:
        """Map a reasoning effort string to a sampling temperature.

        Parameters
        ----------
        reasoning_effort:
            One of ``"low"``, ``"medium"``, or ``"high"``.  Unknown values
            fall back to the ``"medium"`` temperature (``0.5``).

        Returns
        -------
        float
            Sampling temperature to send to Ollama.
        """
        return _EFFORT_TEMPERATURE.get(reasoning_effort, _EFFORT_TEMPERATURE["medium"])

    # ------------------------------------------------------------------
    # Bonus: model discovery
    # ------------------------------------------------------------------

    def get_model_list(self) -> list[str]:
        """Return a list of model names available on this Ollama server.

        Issues a GET request to ``/api/tags``.  Any network or parsing error
        is silently caught and an empty list is returned so that callers can
        degrade gracefully.

        Returns
        -------
        list[str]
            Model name strings (e.g. ``["llama3.2:latest", "qwen3.5:7b"]``),
            or ``[]`` on any error.
        """
        try:
            url = f"{self.base_url}/api/tags"
            headers: dict[str, str] = {}
            if self.api_key is not None:
                headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data: dict = response.json()
            models = data.get("models", [])
            return [m["name"] for m in models if isinstance(m, dict) and "name" in m]
        except Exception:
            logger.debug("get_model_list() failed — returning empty list", exc_info=True)
            return []
