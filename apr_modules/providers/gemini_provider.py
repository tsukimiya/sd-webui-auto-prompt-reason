"""
Gemini provider implementation for sd-webui-auto-prompt-reason.

Uses the Google Generative Language REST API directly (no SDK dependency).
API key is passed as a URL query parameter ``?key=...``.

Notes
-----
* Does **not** use the ``google-generativeai`` SDK to avoid the ~50 MB
  transitive dependency footprint.
* Gemini is **not** OpenAI-compatible; the request / response shape is
  entirely different.
* Thinking budget controls the depth of chain-of-thought reasoning:
  low=1000, medium=5000, high=10000 tokens.
"""

from __future__ import annotations

from typing import Optional

from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import BaseProvider, ProviderType, SecretStr


class GeminiProvider(BaseProvider):
    """Provider implementation for Google Gemini via the REST API.

    Parameters
    ----------
    api_key:
        Required Gemini API key wrapped in :class:`SecretStr`.
    model_name:
        Gemini model identifier (default ``"gemini-2.0-flash"``).
    base_url:
        Root URL for the Generative Language API
        (default ``"https://generativelanguage.googleapis.com/v1"``).
    timeout:
        ``(connect_timeout_s, read_timeout_s)`` tuple (default ``(10, 180)``).
    """

    def __init__(
        self,
        api_key: SecretStr,
        model_name: str = "gemini-2.0-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1",
        timeout: tuple = (10, 180),
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Raise :py:exc:`ValueError` when the API key is missing."""
        if self.api_key is None:
            raise ValueError("api_key is required for Gemini")

    def get_provider_type(self) -> ProviderType:
        """Return :attr:`ProviderType.GEMINI`."""
        return ProviderType.GEMINI

    def supports_reasoning(self) -> bool:
        """Gemini thinking models expose chain-of-thought output."""
        return True

    def build_request(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Construct the Gemini ``generateContent`` request body.

        Parameters
        ----------
        prompt:
            User text prompt.
        image_data:
            Optional base-64-encoded JPEG image for vision requests.
        reasoning_effort:
            ``"low"``, ``"medium"``, or ``"high"`` — maps to a thinking
            budget token count.
        system_prompt:
            Optional system-level instruction.  When non-empty, a top-level
            ``systemInstruction`` field is added with a single ``parts[0].text``
            entry.

        Returns
        -------
        dict
            JSON-serialisable request body for the Gemini REST API.
        """
        thinking_budget = self._effort_to_thinking_budget(reasoning_effort)

        parts: list[dict] = [{"text": prompt}]
        if image_data is not None:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": image_data,
                    }
                }
            )

        payload: dict = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": 1.0,
                "thinkingConfig": {
                    "thinkingBudget": thinking_budget,
                },
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        return payload

    def parse_response(self, raw_response: dict) -> ReasoningResponse:
        """Parse a Gemini ``generateContent`` response into a :class:`ReasoningResponse`.

        Parameters
        ----------
        raw_response:
            Deserialised JSON body returned by the Gemini API.

        Returns
        -------
        ReasoningResponse
            Provider-agnostic representation of the model output.

        Raises
        ------
        ValueError
            If the response contains no candidates.
        """
        candidates = raw_response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini response has no candidates")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        # Concatenate all text parts into a single string.
        full_text = "".join(p.get("text", "") for p in parts)

        # Support <think> tags (used by some Gemini model variants).
        thinking_content: Optional[str] = None
        think_start = full_text.find("<think>")
        think_end = full_text.find("</think>")
        if think_start != -1:
            if think_end != -1:
                thinking_content = full_text[think_start + 7 : think_end]
                final_answer = full_text[think_end + 8 :].strip()
            else:
                thinking_content = full_text[think_start + 7 :]
                final_answer = ""
        else:
            final_answer = full_text.strip()

        # Token usage metadata.
        usage = raw_response.get("usageMetadata", {})
        completion_tokens: int = usage.get("candidatesTokenCount", 0)
        reasoning_tokens_count: Optional[int] = usage.get("thoughtsTokenCount", None)
        total_tokens: int = (
            usage.get("promptTokenCount", 0)
            + completion_tokens
            + (reasoning_tokens_count or 0)
        )

        return ReasoningResponse(
            final_answer=final_answer,
            thinking_content=thinking_content,
            reasoning_tokens=reasoning_tokens_count,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_time_ms=0,
            model=self.model_name,
            provider="gemini",
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_api_url(self) -> str:
        """Return the fully-qualified Gemini ``generateContent`` endpoint URL.

        The API key is embedded as a ``?key=`` query parameter per the
        Generative Language API conventions.
        """
        return (
            f"{self.base_url}/models/{self.model_name}"
            f":generateContent?key={self.api_key.get_secret_value()}"  # type: ignore[union-attr]
        )

    @staticmethod
    def _effort_to_thinking_budget(reasoning_effort: str) -> int:
        """Map a reasoning effort string to a Gemini thinking budget (tokens).

        Parameters
        ----------
        reasoning_effort:
            One of ``"low"``, ``"medium"``, ``"high"``. Unknown values fall
            back to the medium budget.

        Returns
        -------
        int
            Token budget for the ``thinkingConfig.thinkingBudget`` field.
        """
        mapping = {
            "low": 1000,
            "medium": 5000,
            "high": 10000,
        }
        return mapping.get(reasoning_effort, 5000)
