"""
LLM client integration layer for sd-webui-auto-prompt-reason.

This module provides :class:`LLMClient`, the unified entry point for making LLM
API calls.  It uses a factory pattern to create the correct provider instance,
dispatches the HTTP request on a background thread when called asynchronously,
measures wall-clock timing, and returns a :class:`~modules.models.ReasoningResponse`.

Design notes
------------
* No SD WebUI or Gradio imports are allowed here.
* All blocking HTTP calls in :py:meth:`LLMClient.generate_async` run on daemon
  threads so the SD WebUI main thread is never blocked.
* Timing is injected via :func:`dataclasses.replace` rather than mutating the
  returned dataclass in-place (safe for frozen or plain dataclasses alike).
* Gemini uses a ``?key=`` URL query parameter instead of a ``Bearer`` token
  header — the :py:meth:`_build_headers` helper omits ``Authorization`` for
  that provider.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Optional

import requests

from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import BaseProvider, ProviderType
from apr_modules.providers.gemini_provider import GeminiProvider
from apr_modules.providers.ollama_provider import OllamaProvider
from apr_modules.providers.openai_provider import OpenAICompatibleProvider


class LLMClient:
    """Unified client for dispatching LLM API calls across multiple providers.

    Use :py:meth:`create` to instantiate via the factory pattern when you have
    only a string identifier for the desired provider.  Alternatively, build a
    concrete :class:`~modules.providers.base_provider.BaseProvider` yourself and
    pass it directly to ``__init__``.

    Parameters
    ----------
    provider:
        A fully-configured :class:`~modules.providers.base_provider.BaseProvider`
        instance.  The client owns no configuration state of its own; all
        provider-specific logic (URL, headers, payload structure) is delegated
        to this object.

    Examples
    --------
    >>> from modules.providers.base_provider import SecretStr
    >>> client = LLMClient.create(
    ...     "ollama",
    ...     base_url="http://localhost:11434",
    ...     model_name="qwen3.5:7b",
    ... )
    >>> response = client.generate("Describe this image in one sentence.")
    """

    def __init__(self, provider: BaseProvider) -> None:
        """Initialise the client with a pre-built provider instance.

        Parameters
        ----------
        provider:
            A concrete :class:`~modules.providers.base_provider.BaseProvider`
            subclass instance ready to make API calls.
        """
        self._provider: BaseProvider = provider
        self._logger: logging.Logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, provider_type: str, **kwargs: object) -> "LLMClient":
        """Instantiate a :class:`LLMClient` from a provider type string.

        Parameters
        ----------
        provider_type:
            One of ``"ollama"``, ``"openai_compatible"``, or ``"gemini"``.
        **kwargs:
            Constructor keyword arguments forwarded verbatim to the matching
            provider class.

        Returns
        -------
        LLMClient
            A ready-to-use client wrapping the requested provider.

        Raises
        ------
        ValueError
            If *provider_type* is not a recognised value.

        Examples
        --------
        >>> client = LLMClient.create("ollama", model_name="llama3.2:latest")
        >>> client = LLMClient.create(
        ...     "openai_compatible",
        ...     base_url="http://localhost:1234/v1",
        ...     model_name="mistral-7b",
        ... )
        """
        if provider_type == "ollama":
            provider: BaseProvider = OllamaProvider(**kwargs)  # type: ignore[arg-type]
        elif provider_type == "openai_compatible":
            provider = OpenAICompatibleProvider(**kwargs)  # type: ignore[arg-type]
        elif provider_type == "gemini":
            provider = GeminiProvider(**kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(
                f"Unknown provider type {provider_type!r}. "
                "Expected one of: 'ollama', 'openai_compatible', 'gemini'."
            )
        return cls(provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> ReasoningResponse:
        """Send a prompt to the configured LLM provider and return the response.

        This method runs **synchronously** on the calling thread.  For
        non-blocking usage from a UI context, prefer
        :py:meth:`generate_async`.

        Parameters
        ----------
        prompt:
            The user-facing text prompt to send to the model.
        image_data:
            Optional base-64-encoded image string for vision-capable models.
            Pass ``None`` when no image is involved.
        reasoning_effort:
            Hint for models that support variable reasoning depth.
            Accepted values: ``"low"``, ``"medium"``, ``"high"``.
        system_prompt:
            Optional system-level instruction prepended to the conversation
            before the user turn.  Pass ``None`` to omit the system message
            and rely on the provider's built-in defaults.

        Returns
        -------
        ReasoningResponse
            Provider-agnostic response including the final answer, optional
            thinking content, token counts, and elapsed wall-clock time.

        Raises
        ------
        ValueError
            Propagated from :py:meth:`~BaseProvider.validate_config` when the
            provider is misconfigured.
        requests.HTTPError
            When the server returns a non-2xx HTTP status code.
        requests.RequestException
            On any lower-level network error (timeout, connection refused, etc.).
        """
        self._provider.validate_config()

        payload = self._provider.build_request(
            prompt,
            image_data,
            reasoning_effort,
            system_prompt,
        )
        url = self._build_url()
        headers = self._build_headers()

        self._logger.debug(
            "Sending request to %s (provider=%s, model=%s)",
            url,
            self._provider.get_provider_type().value,
            self._provider.model_name,
        )

        start = time.time()
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self._provider.timeout,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        response.raise_for_status()

        response_obj = self._provider.parse_response(response.json())

        self._logger.debug(
            "Request completed in %d ms (provider=%s)",
            elapsed_ms,
            self._provider.get_provider_type().value,
        )

        # Inject wall-clock timing without mutating the dataclass in-place.
        return dataclasses.replace(response_obj, reasoning_time_ms=elapsed_ms)

    def generate_async(
        self,
        prompt: str,
        callback: Callable[[ReasoningResponse], None],
        error_callback: Callable[[Exception], None],
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> threading.Thread:
        """Send a prompt on a background daemon thread.

        The calling thread is never blocked.  On completion, *callback* is
        invoked with the :class:`ReasoningResponse`.  On any exception,
        *error_callback* is invoked with the exception instance.

        Parameters
        ----------
        prompt:
            The user-facing text prompt.
        callback:
            Invoked on the worker thread with the successful
            :class:`ReasoningResponse`.
        error_callback:
            Invoked on the worker thread with the :py:exc:`Exception` raised
            during the request.
        image_data:
            Optional base-64-encoded image string.
        reasoning_effort:
            ``"low"``, ``"medium"``, or ``"high"``.
        system_prompt:
            Optional system-level instruction.  Pass ``None`` to omit.

        Returns
        -------
        threading.Thread
            The already-started daemon thread executing the request.
        """

        def _worker() -> None:
            try:
                result = self.generate(
                    prompt,
                    image_data=image_data,
                    reasoning_effort=reasoning_effort,
                    system_prompt=system_prompt,
                )
                callback(result)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "generate_async worker raised an exception: %s",
                    exc,
                    exc_info=True,
                )
                error_callback(exc)

        thread = threading.Thread(target=_worker, name="llm-client-worker")
        thread.daemon = True
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        """Construct the full endpoint URL for the configured provider.

        Returns
        -------
        str
            The POST target URL, including any required query parameters (e.g.
            the Gemini ``?key=`` param).
        """
        provider_type = self._provider.get_provider_type()

        if provider_type == ProviderType.OLLAMA:
            return f"{self._provider.base_url}/api/chat"

        if provider_type == ProviderType.GEMINI:
            # GeminiProvider._get_api_url() embeds the API key as a query param.
            gemini_provider: GeminiProvider = self._provider  # type: ignore[assignment]
            return gemini_provider._get_api_url()

        # OpenAI-compatible: base_url may already include /v1.
        base = self._provider.base_url
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _build_headers(self) -> dict:
        """Return the HTTP request headers for the current provider.

        Always includes ``Content-Type: application/json``.  An
        ``Authorization: Bearer …`` header is appended when
        :attr:`~BaseProvider.api_key` is set, **except** for Gemini which
        embeds its key in the URL query parameter instead.

        Returns
        -------
        dict
            Header mapping safe to pass to :func:`requests.post`.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}

        # Gemini uses ?key= in the URL — do NOT add an Authorization header.
        if isinstance(self._provider, GeminiProvider):
            return headers

        if self._provider.api_key is not None:
            # Unwrap the secret only here; never log or store the plaintext.
            headers["Authorization"] = (
                f"Bearer {self._provider.api_key.get_secret_value()}"
            )

        return headers
