"""
Ollama provider implementation for sd-webui-auto-prompt-reason.

This module implements the :class:`OllamaProvider` concrete class that talks to
a locally-running Ollama inference server (default: ``http://localhost:11434``).

Response parsing
----------------
Two Ollama response shapes are supported:

* **Shape A** (Ollama ≥ 0.7.0): the ``message`` object contains a dedicated
  ``thinking`` field alongside ``content``.
* **Shape B** (Ollama < 0.7.0): chain-of-thought is embedded in ``content``
  between ``<think>`` / ``</think>`` tags.

The ``</think>`` tag may be absent when the server truncates the response
(e.g. due to a token limit).  In that case everything after ``<think>`` is
treated as thinking and ``final_answer`` is set to an empty string.

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
            "keep_alive": "10m",
            "options": {
                "temperature": self._effort_to_temperature(reasoning_effort),
                "num_ctx": 8192,
            },
        }

    def parse_response(self, raw_response: dict) -> ReasoningResponse:
        """Parse an Ollama native ``/api/chat`` response into a :class:`ReasoningResponse`.

        Handles both Shape A (``thinking`` field) and Shape B (``<think>``
        tags embedded in ``content``).

        Parameters
        ----------
        raw_response:
            Deserialised JSON body from the Ollama ``/api/chat`` endpoint.

        Returns
        -------
        ReasoningResponse
            Normalised provider-agnostic response.
        """
        # --- 診断ログ: 生レスポンス構造の確認 ---
        # ネイティブ形式: message.content / message.thinking
        message: dict[str, Any] = raw_response.get("message", {})
        content: str = message.get("content", "")
        thinking: Optional[str] = message.get("thinking")
        logger.info(
            "parse_response: raw keys=%s message=%s content_len=%d"
            " thinking_len=%s done_reason=%r",
            sorted(raw_response.keys()) if isinstance(raw_response, dict) else type(raw_response).__name__,
            message,
            len(content),
            len(thinking) if thinking else "None",
            raw_response.get("done_reason", "N/A"),
        )

        thinking_content: Optional[str] = None
        final_answer: str = content
        _shape_detected: str = "plain(no thinking)"

        # --- Shape A: Ollama ≥ 0.7.0 has a dedicated ``thinking`` field ---
        if "thinking" in message:
            _shape_detected = "A(dedicated thinking field)"
            thinking_content = message["thinking"] or None
            final_answer = content
        else:
            # --- Shape B: thinking embedded in <think>…</think> tags ---
            think_start = content.find("<think>")
            think_end = content.find("</think>")

            if think_start != -1 and think_end != -1:
                # Both tags present — clean extraction
                _shape_detected = "B(both think tags)"
                thinking_content = content[think_start + 7 : think_end]
                final_answer = content[think_end + 8 :].strip()
            elif think_start != -1:
                # Opening tag only — response was truncated
                _shape_detected = "B(unclosed think tag — truncated)"
                thinking_content = content[think_start + 7 :]
                final_answer = ""
            # else: no tags — keep thinking_content=None, final_answer=content

        # ネイティブレスポンスのトークンカウント
        eval_count: int = raw_response.get("eval_count", 0)
        prompt_eval_count: int = raw_response.get("prompt_eval_count", 0)

        # --- 診断ログ: パース結果の確認 ---
        logger.info(
            "parse_response: final_answer_len=%d thinking_len=%s"
            " shape=%s eval_count=%d prompt_eval_count=%d",
            len(final_answer.strip()),
            len(thinking_content) if thinking_content else "None",
            _shape_detected,
            eval_count,
            prompt_eval_count,
        )

        return ReasoningResponse(
            final_answer=final_answer.strip(),
            thinking_content=thinking_content or None,
            reasoning_tokens=(
                len(thinking_content.split()) if thinking_content else None
            ),
            completion_tokens=eval_count,
            total_tokens=eval_count + prompt_eval_count,
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
