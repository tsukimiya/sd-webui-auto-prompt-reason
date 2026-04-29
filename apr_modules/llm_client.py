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

Logging
-------
Each call to :py:meth:`generate` emits DEBUG-level log lines at the following
lifecycle stages so that extension logs make the request path immediately
visible:

1. ``[CONFIG VALIDATED]``  — provider validated, config is intact.
2. ``[PAYLOAD PREPARED]``  — request body built; logged *without* raw image data.
3. ``[REQUEST SENDING]``   — URL, provider, model, timeout logged just before
   ``requests.post`` is called.  Gemini ``?key=`` is redacted.
   ``Authorization`` header value is never logged.
4. ``[RESPONSE RECEIVED]`` — HTTP status code and elapsed time logged.
5. ``[JSON PARSED]``       — top-level keys of the parsed JSON body logged.
6. ``[REQUEST TIMEOUT]``   — explicit timeout log including URL and timeout.
7. ``[REQUEST FAILED]``    — status code (if HTTP) or exception type for any
   failure; also logged from the async worker wrapper.

None of these messages include raw API keys, raw Authorization header values,
or full base-64 image blobs.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from typing import Any, Callable, Optional

import requests

from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import BaseProvider, ProviderType
from apr_modules.providers.gemini_provider import GeminiProvider
from apr_modules.providers.ollama_provider import OllamaProvider
from apr_modules.providers.openai_provider import OpenAICompatibleProvider

# ---------------------------------------------------------------------------
# Module-level redaction helpers
# ---------------------------------------------------------------------------

# Matches the ?key=<value> query parameter used by Gemini.
_KEY_PARAM_RE = re.compile(r"(\?key=)[^&\s]+")
# Maximum characters shown for a base-64 blob before truncation.
_B64_PREVIEW_LEN = 16


def _redact_url(url: str) -> str:
    """Return *url* with any ``?key=<value>`` query parameter redacted.

    Parameters
    ----------
    url:
        The raw URL that may contain a Gemini ``?key=`` param.

    Returns
    -------
    str
        URL safe to include in log output.

    Examples
    --------
    >>> _redact_url("https://api.example.com/v1/models/foo:gen?key=secret123")
    'https://api.example.com/v1/models/foo:gen?key=**redacted**'
    >>> _redact_url("http://localhost:11434/api/chat")
    'http://localhost:11434/api/chat'
    """
    return _KEY_PARAM_RE.sub(r"\g<1>**redacted**", url)


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with ``Authorization`` value masked.

    Parameters
    ----------
    headers:
        Raw HTTP headers dict that may contain credentials.

    Returns
    -------
    dict[str, str]
        Safe-to-log copy of the headers.
    """
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            redacted[key] = "Bearer **redacted**"
        else:
            redacted[key] = value
    return redacted


def _summarise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a log-safe summary of *payload* without full image blobs.

    Recursively truncates any string value longer than :data:`_B64_PREVIEW_LEN`
    characters inside ``inlineData.data`` (Gemini) or OpenAI-style image-url
    ``data:image/…;base64,…`` values, replacing the remainder with
    ``…<N bytes>``.

    Parameters
    ----------
    payload:
        The raw request body dict.

    Returns
    -------
    dict[str, Any]
        A shallow-copied dict safe to pass to a logger.
    """
    import copy

    summary = copy.deepcopy(payload)
    _truncate_images_inplace(summary)
    return summary


def _truncate_images_inplace(obj: Any) -> None:  # noqa: ANN401
    """Recursively walk *obj* and truncate long base-64 strings in-place."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "data" and isinstance(value, str) and len(value) > _B64_PREVIEW_LEN:
                # Gemini inlineData.data or any other raw b64 field.
                obj[key] = value[:_B64_PREVIEW_LEN] + f"…<{len(value)} chars total>"
            elif key == "url" and isinstance(value, str) and value.startswith("data:"):
                # OpenAI-style data-URL: data:image/jpeg;base64,<blob>
                obj[key] = value[:_B64_PREVIEW_LEN] + f"…<{len(value)} chars total>"
            else:
                _truncate_images_inplace(value)
    elif isinstance(obj, list):
        for item in obj:
            _truncate_images_inplace(item)


def _payload_meta(payload: dict[str, Any]) -> str:
    """Return a compact one-line metadata string describing *payload* shape.

    Reports: top-level keys, whether a system instruction is present, whether
    any image data is embedded, and message count where applicable.

    Parameters
    ----------
    payload:
        Raw (unredacted) request payload.

    Returns
    -------
    str
        Log-friendly summary string, e.g.
        ``"keys=['model','messages'] messages=2 has_image=True"``.
    """
    parts: list[str] = [f"keys={sorted(payload.keys())}"]

    # Message count (Ollama / OpenAI-compatible).
    if "messages" in payload:
        parts.append(f"messages={len(payload['messages'])}")

    # Gemini-style system instruction.
    if "systemInstruction" in payload:
        parts.append("has_system=True")

    # Check for embedded images anywhere in the payload.
    has_image = _contains_image(payload)
    if has_image:
        parts.append("has_image=True")

    return " ".join(parts)


def _contains_image(obj: Any) -> bool:  # noqa: ANN401
    """Return ``True`` if *obj* contains any embedded image data."""
    if isinstance(obj, dict):
        # Gemini inlineData.
        if "inlineData" in obj:
            return True
        # OpenAI data-URL.
        if obj.get("type") == "image_url":
            return True
        return any(_contains_image(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_image(item) for item in obj)
    return False


def _accumulate_ollama_stream(response: requests.Response) -> dict[str, Any]:
    """Accumulate an Ollama ``/api/chat`` streaming response into a single dict.

    Each line from the response is a JSON object (NDJSON).  We walk through
    the stream, concatenating ``message.content`` and ``message.thinking``
    chunks.  When ``done: true`` is seen we return a dict that mimics the
    shape of a non-streaming response so that the existing
    ``parse_response()`` logic can be reused without modification.

    Parameters
    ----------
    response:
        The raw :class:`requests.Response` from the streaming POST.

    Returns
    -------
    dict[str, Any]
        A dict with the same top-level keys as a non-streaming Ollama
        response, including ``message`` (with ``content`` and optionally
        ``thinking``), ``eval_count``, ``prompt_eval_count``, etc.
    """
    import json

    content_parts: list[str] = []
    thinking_parts: list[str] = []
    final_data: dict[str, Any] = {}
    stream_completed = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg: dict[str, Any] = data.get("message", {})
        if "content" in msg:
            content_parts.append(msg["content"])
        if "thinking" in msg and msg["thinking"]:
            thinking_parts.append(msg["thinking"])

        if data.get("done"):
            final_data = data
            stream_completed = True
            break

    if not stream_completed:
        logger = logging.getLogger(__name__)
        logger.warning(
            "[STREAM INCOMPLETE] Ollama stream ended without done=true — "
            "using accumulated content_len=%d thinking_len=%s",
            len("".join(content_parts)),
            len("".join(thinking_parts)) if thinking_parts else "None",
        )

    # Build a synthetic non-streaming response.
    accumulated_message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if thinking_parts:
        accumulated_message["thinking"] = "".join(thinking_parts)

    final_data["message"] = accumulated_message
    return final_data


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


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
        provider_name = self._provider.get_provider_type().value
        model_name = self._provider.model_name

        # ------------------------------------------------------------------
        # Stage 1: config validated
        # ------------------------------------------------------------------
        self._provider.validate_config()
        self._logger.info(
            "[CONFIG VALIDATED] provider=%s model=%s timeout=%s",
            provider_name,
            model_name,
            self._provider.timeout,
        )

        # ------------------------------------------------------------------
        # Stage 2: payload prepared
        # ------------------------------------------------------------------
        payload = self._provider.build_request(
            prompt,
            image_data,
            reasoning_effort,
            system_prompt,
        )
        self._logger.info(
            "[PAYLOAD PREPARED] provider=%s model=%s effort=%s "
            "has_prompt=%s has_system=%s has_image=%s payload_meta=(%s)",
            provider_name,
            model_name,
            reasoning_effort,
            bool(prompt),
            system_prompt is not None,
            image_data is not None,
            _payload_meta(payload),
        )

        url = self._build_url()
        headers = self._build_headers()

        # ------------------------------------------------------------------
        # Stage 3: request about to send
        # ------------------------------------------------------------------
        self._logger.info(
            "[REQUEST SENDING] provider=%s model=%s url=%s headers=%s timeout=%s",
            provider_name,
            model_name,
            _redact_url(url),
            _redact_headers(headers),
            self._provider.timeout,
        )

        start = time.time()
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._provider.timeout,
            )
            elapsed_ms = int((time.time() - start) * 1000)

            # ------------------------------------------------------------------
            # Stage 4: response received
            # ------------------------------------------------------------------
            self._logger.info(
                "[RESPONSE RECEIVED] provider=%s model=%s status=%d elapsed_ms=%d",
                provider_name,
                model_name,
                response.status_code,
                elapsed_ms,
            )

            response.raise_for_status()

        except requests.HTTPError as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            status_code: Optional[int] = (
                exc.response.status_code if exc.response is not None else None
            )
            # レスポンスボディをログ出力 — 524等のエラー原因特定に不可欠
            response_body = ""
            if exc.response is not None:
                try:
                    response_body = exc.response.text[:500]
                except Exception:  # noqa: BLE001
                    response_body = "<unable to read response body>"
            self._logger.warning(
                "[REQUEST FAILED] provider=%s model=%s reason=http_error "
                "status=%s elapsed_ms=%d response_body=%.200s error=%s",
                provider_name,
                model_name,
                status_code,
                elapsed_ms,
                response_body,
                exc,
            )
            raise
        except requests.ReadTimeout as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            self._logger.warning(
                "[REQUEST TIMEOUT] provider=%s model=%s url=%s timeout=%s "
                "elapsed_ms=%d error=%s",
                provider_name,
                model_name,
                _redact_url(url),
                self._provider.timeout,
                elapsed_ms,
                exc,
            )
            self._logger.debug(
                "[REQUEST FAILED] provider=%s model=%s reason=ReadTimeout "
                "url=%s timeout=%s elapsed_ms=%d error=%s",
                provider_name,
                model_name,
                _redact_url(url),
                self._provider.timeout,
                elapsed_ms,
                exc,
            )
            raise
        except requests.RequestException as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            self._logger.debug(
                "[REQUEST FAILED] provider=%s model=%s reason=%s "
                "url=%s timeout=%s elapsed_ms=%d error=%s",
                provider_name,
                model_name,
                type(exc).__name__,
                _redact_url(url),
                self._provider.timeout,
                elapsed_ms,
                exc,
            )
            raise

        # ------------------------------------------------------------------
        # Stage 5: JSON parsed
        # ------------------------------------------------------------------
        if provider_name == ProviderType.OLLAMA.value and payload.get("stream"):
            raw_json = _accumulate_ollama_stream(response)
            self._logger.info(
                "[STREAM ACCUMULATED] provider=%s model=%s content_len=%d thinking_len=%s",
                provider_name,
                model_name,
                len(raw_json.get("message", {}).get("content", "")),
                len(raw_json.get("message", {}).get("thinking", "")) or "None",
            )
        else:
            raw_json = response.json()

        self._logger.info(
            "[JSON PARSED] provider=%s model=%s response_keys=%s",
            provider_name,
            model_name,
            sorted(raw_json.keys()) if isinstance(raw_json, dict) else type(raw_json).__name__,
        )

        response_obj = self._provider.parse_response(raw_json)

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
            provider_name = self._provider.get_provider_type().value
            model_name = self._provider.model_name
            self._logger.debug(
                "[ASYNC WORKER STARTED] provider=%s model=%s",
                provider_name,
                model_name,
            )
            try:
                result = self.generate(
                    prompt,
                    image_data=image_data,
                    reasoning_effort=reasoning_effort,
                    system_prompt=system_prompt,
                )
                self._logger.debug(
                    "[ASYNC WORKER DONE] provider=%s model=%s elapsed_ms=%d",
                    provider_name,
                    model_name,
                    result.reasoning_time_ms,
                )
                callback(result)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "[REQUEST FAILED] async_worker provider=%s model=%s "
                    "reason=%s error=%s",
                    provider_name,
                    model_name,
                    type(exc).__name__,
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
