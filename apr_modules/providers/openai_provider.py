"""
OpenAI-compatible provider for sd-webui-auto-prompt-reason.

Supports any server exposing an OpenAI-compatible ``/v1/chat/completions``
endpoint: LM Studio, vLLM, OpenAI itself, DeepSeek, etc.

Security note
-------------
The ``api_key`` is stored as :class:`SecretStr` and is unwrapped *only*
inside :py:meth:`_build_headers`. It is never logged or stored in any other
form.
"""

from __future__ import annotations

from typing import Optional

from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import BaseProvider, ProviderType, SecretStr


class OpenAICompatibleProvider(BaseProvider):
    """Provider for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Parameters
    ----------
    base_url:
        Root URL of the API server, e.g. ``"http://localhost:1234/v1"``.
        Trailing slashes are stripped automatically.
    model_name:
        Model identifier to pass in the ``"model"`` field of each request.
        Must be non-empty; validation is enforced in :py:meth:`validate_config`.
    api_key:
        Optional bearer token wrapped in :class:`SecretStr`. Pass ``None``
        for unauthenticated local servers.
    timeout:
        ``(connect_timeout_s, read_timeout_s)`` tuple. Defaults to
        ``(10, 180)``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "",
        api_key: Optional[SecretStr] = None,
        timeout: tuple = (10, 180),
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Raise :py:exc:`ValueError` when ``model_name`` is empty."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")

    def get_provider_type(self) -> ProviderType:
        """Return :attr:`ProviderType.OPENAI_COMPATIBLE`."""
        return ProviderType.OPENAI_COMPATIBLE

    def supports_reasoning(self) -> bool:
        """Return ``True`` — handles ``<think>`` tags and ``reasoning_content`` fields.

        Even if the underlying model does not produce explicit reasoning, this
        provider parses both formats gracefully and returns ``None`` for
        ``thinking_content`` when neither is present.
        """
        return True

    def build_request(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Build the JSON payload for ``POST /v1/chat/completions``.

        Parameters
        ----------
        prompt:
            User-facing text prompt.
        image_data:
            Optional base-64-encoded JPEG image. When provided, the message
            content is structured as a vision array.
        reasoning_effort:
            Controls the ``temperature`` via :py:meth:`_effort_to_temperature`.
            Accepted values: ``"low"``, ``"medium"``, ``"high"``.
        system_prompt:
            Optional system-level instruction.  When non-empty, a
            ``{"role": "system", …}`` message is prepended to ``messages``.

        Returns
        -------
        dict
            JSON-serialisable request payload.
        """
        if image_data is not None:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        else:
            content = prompt  # type: ignore[assignment]

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": self._effort_to_temperature(reasoning_effort),
        }

    def parse_response(self, raw_response: dict) -> ReasoningResponse:
        """Normalise a raw ``/v1/chat/completions`` response.

        Handles three response shapes:

        1. ``choices[0].message.reasoning_content`` — DeepSeek R1 and similar
           providers that expose a dedicated reasoning field.
        2. ``<think>…</think>`` tags embedded in ``choices[0].message.content``
           — used by some models that inline their chain-of-thought.
        3. Plain content with no reasoning markers — returned as-is with
           ``thinking_content=None``.

        Parameters
        ----------
        raw_response:
            Deserialised JSON from the provider.

        Returns
        -------
        ReasoningResponse
            Provider-agnostic response object.
        """
        choices = raw_response.get("choices", [{}])
        message = choices[0].get("message", {}) if choices else {}
        content: str = message.get("content") or ""

        # 1. Check dedicated reasoning_content field (DeepSeek R1, etc.)
        thinking_content: Optional[str] = message.get("reasoning_content") or None

        # 2. Fall back to <think> tag parsing
        if not thinking_content:
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1:
                if think_end != -1:
                    thinking_content = content[think_start + 7:think_end]
                    content = content[think_end + 8:].strip()
                else:
                    # Truncated response — capture partial thinking, clear content
                    thinking_content = content[think_start + 7:]
                    content = ""

        usage = raw_response.get("usage", {})
        completion_tokens: int = usage.get("completion_tokens", 0)
        total_tokens: int = usage.get("total_tokens", completion_tokens)

        return ReasoningResponse(
            final_answer=content.strip(),
            thinking_content=thinking_content if thinking_content else None,
            reasoning_tokens=None,  # not always available in OpenAI-compat APIs
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_time_ms=0,
            model=self.model_name,
            provider="openai_compatible",
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        """Return HTTP headers for the request.

        Always includes ``Content-Type: application/json``. Appends a
        ``Bearer`` ``Authorization`` header when an ``api_key`` is set.

        Returns
        -------
        dict
            Header mapping safe to pass to any HTTP client.
        """
        headers: dict = {"Content-Type": "application/json"}
        if self.api_key is not None:
            # Unwrap the secret only here — never store or log the plaintext.
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _effort_to_temperature(self, effort: str) -> float:
        """Map a reasoning-effort label to a sampling temperature.

        Parameters
        ----------
        effort:
            One of ``"low"``, ``"medium"``, or ``"high"``. Unknown values
            fall back to ``0.5``.

        Returns
        -------
        float
            Sampling temperature in the range ``[0.0, 1.0]``.
        """
        _map: dict[str, float] = {
            "low": 0.8,
            "medium": 0.5,
            "high": 0.2,
        }
        return _map.get(effort, 0.5)
