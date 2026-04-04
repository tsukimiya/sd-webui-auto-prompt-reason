"""
Base provider interface for sd-webui-auto-prompt-reason.

This module defines the abstract base class that all LLM provider implementations
must conform to. It provides the contract for building requests, parsing responses,
and managing provider-specific configuration.

Providers are consumed by ``modules/llm_client.py`` via a factory pattern. No
concrete dispatch logic (``if provider == "ollama" ...``) lives here; that
responsibility belongs to the factory.

Security note
-------------
API keys are stored exclusively as :class:`SecretStr` instances. The class
intentionally masks its value in ``__repr__`` / ``__str__`` so that accidental
log or debug output never leaks credentials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from apr_modules.models.reasoning_response import ReasoningResponse


# ---------------------------------------------------------------------------
# SecretStr — opaque credential wrapper
# ---------------------------------------------------------------------------


class SecretStr:
    """Immutable string wrapper that prevents accidental credential exposure.

    The raw value is accessible only via :py:meth:`get_secret_value` so callers
    are forced to be explicit about wanting the plaintext credential.

    Examples
    --------
    >>> key = SecretStr("sk-supersecret")
    >>> repr(key)
    "SecretStr('**********')"
    >>> str(key)
    '**********'
    >>> key.get_secret_value()
    'sk-supersecret'
    >>> SecretStr("abc") == SecretStr("abc")
    True
    """

    _MASK = "**********"

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"SecretStr requires a str, got {type(value).__name__!r}")
        # Store under a name-mangled attribute to reduce accidental access.
        self.__secret_value: str = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_secret_value(self) -> str:
        """Return the plaintext secret value.

        Callers should avoid assigning the result to variables that may appear
        in log output.
        """
        return self.__secret_value

    # ------------------------------------------------------------------
    # Dunder methods — all masked
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401
        """Return a masked representation safe for logs."""
        return f"SecretStr('{self._MASK}')"

    def __str__(self) -> str:
        """Return the masked string value."""
        return self._MASK

    def __eq__(self, other: object) -> bool:
        """Compare equality by the underlying secret value."""
        if isinstance(other, SecretStr):
            return self.__secret_value == other.__secret_value  # type: ignore[attr-defined]
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.__secret_value)


# ---------------------------------------------------------------------------
# ProviderType — enumeration of supported provider kinds
# ---------------------------------------------------------------------------


class ProviderType(Enum):
    """Enumeration of the LLM provider back-ends supported by this extension.

    Each concrete :class:`BaseProvider` subclass must declare which type it
    represents by implementing :py:meth:`BaseProvider.get_provider_type`.

    Members
    -------
    OLLAMA
        Local Ollama inference server (http://localhost:11434 by default).
    OPENAI_COMPATIBLE
        Any server that exposes an OpenAI-compatible ``/v1/chat/completions``
        endpoint (e.g. LM Studio, vLLM, OpenAI itself).
    GEMINI
        Google Generative Language / Gemini API.
    CUSTOM
        Escape hatch for third-party or experimental providers.
    """

    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    GEMINI = "gemini"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# BaseProvider — abstract base class
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract base class for all LLM provider implementations.

    Subclasses **must** implement every ``@abstractmethod`` defined here.
    Attempting to instantiate :class:`BaseProvider` directly raises
    :py:exc:`TypeError`.

    Parameters
    ----------
    base_url:
        Root URL of the provider API endpoint, e.g.
        ``"http://localhost:11434"``.
    api_key:
        Optional credential. Pass ``None`` for providers that do not
        require authentication (e.g. local Ollama).
    model_name:
        Identifier of the model to invoke, e.g. ``"llama3.2:latest"`` or
        ``"gemini-2.0-flash-thinking-exp"``.
    timeout:
        ``(connect_timeout, read_timeout)`` in seconds passed to the
        underlying HTTP client. Defaults to ``(10, 60)``.

    Notes
    -----
    * No SD WebUI or Gradio imports are allowed in this module.
    * Logging of the ``api_key`` value is prevented by the :class:`SecretStr`
      type — never unwrap the key outside of the actual HTTP request call.
    * Forward-reference ``"ReasoningResponse"`` is used instead of a direct
      import to prevent circular dependencies with ``modules.models``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[SecretStr],
        model_name: str,
        timeout: tuple = (10, 60),
    ) -> None:
        """Initialise shared provider state.

        Parameters
        ----------
        base_url:
            Root URL for the provider API (no trailing slash required).
        api_key:
            Wrapped secret credential, or ``None`` for unauthenticated
            providers.
        model_name:
            Model identifier string understood by the target provider.
        timeout:
            Two-element tuple ``(connect_timeout_s, read_timeout_s)``.
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[SecretStr] = api_key
        self.model_name: str = model_name
        self.timeout: tuple = timeout

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def build_request(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Construct the provider-specific HTTP request payload.

        Parameters
        ----------
        prompt:
            The user-facing text prompt to send to the model.
        image_data:
            Optional base-64-encoded image string for vision-capable models.
            Pass ``None`` when no image is being submitted.
        reasoning_effort:
            Hint for models that support variable reasoning depth.
            Typical values are ``"low"``, ``"medium"``, and ``"high"``.
        system_prompt:
            Optional system-level instruction prepended to the conversation
            before the user turn.  Pass ``None`` to omit the system message
            entirely and rely on the provider's built-in defaults.

        Returns
        -------
        dict
            A JSON-serialisable dict ready to be POSTed to the provider
            endpoint. The exact structure is provider-specific.
        """

    @abstractmethod
    def parse_response(self, raw_response: dict) -> "ReasoningResponse":
        """Extract a normalised :class:`~modules.models.ReasoningResponse` from the raw API reply.

        Parameters
        ----------
        raw_response:
            The deserialised JSON body returned by the provider's API.

        Returns
        -------
        ReasoningResponse
            A provider-agnostic representation of the model's output.

        Raises
        ------
        ValueError
            If the response structure is missing expected fields.
        """

    @abstractmethod
    def supports_reasoning(self) -> bool:
        """Return whether this provider/model exposes a dedicated reasoning field.

        Returns
        -------
        bool
            ``True`` if the provider returns an explicit chain-of-thought or
            reasoning block alongside the final answer; ``False`` otherwise.
        """

    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider configuration and raise on any invalid state.

        This method is called before the first request is issued to surface
        misconfiguration early (e.g. missing required ``api_key``, unreachable
        ``base_url``, unsupported ``model_name`` format).

        Raises
        ------
        ValueError
            If any configuration value is missing or invalid.
        """

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return the :class:`ProviderType` enum member for this implementation.

        Returns
        -------
        ProviderType
            The provider type constant that identifies this back-end.
        """

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401
        """Return a developer-friendly representation without exposing secrets."""
        return (
            f"{self.__class__.__name__}("
            f"base_url={self.base_url!r}, "
            f"model_name={self.model_name!r}, "
            f"api_key={'<set>' if self.api_key is not None else '<unset>'}"
            f")"
        )
