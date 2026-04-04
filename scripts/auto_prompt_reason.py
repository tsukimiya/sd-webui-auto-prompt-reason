"""
SD WebUI extension entry point for sd-webui-auto-prompt-reason.

This module integrates with AUTOMATIC1111 SD WebUI's script lifecycle via the
:class:`AutoPromptReason` class.  When enabled, it intercepts the image
generation pipeline, sends the user's prompt to a configured LLM provider, and
injects the LLM-enhanced prompt back into the processing object according to
the ``prompt_injection`` mode set in ``config.yaml``.

Design notes
------------
* All SD WebUI and Gradio imports are guarded with ``try/except`` so the module
  can be imported in test environments where those packages are absent.
* The ``run()`` method is called by SD WebUI on a worker thread — it is safe to
  call ``LLMClient.generate()`` synchronously here.
* No global mutable config state is used; config is loaded once per instance
  and cached via :py:meth:`AutoPromptReason._load_config`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from apr_modules.history_manager import HistoryManager
from apr_modules.llm_client import LLMClient
from apr_modules.models.reasoning_response import ReasoningResponse
from apr_modules.providers.base_provider import SecretStr
from apr_modules.ui_components import build_thinking_display, build_metrics_display, update_thinking_display

# ---------------------------------------------------------------------------
# SD WebUI / Gradio — only available at runtime inside AUTOMATIC1111
# ---------------------------------------------------------------------------

import modules.scripts as scripts
import gradio as gr

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension root — used to locate config.yaml at runtime
# ---------------------------------------------------------------------------

_EXTENSION_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Script class
# ---------------------------------------------------------------------------


class AutoPromptReason(scripts.Script):  # type: ignore[misc,valid-type]
    """SD WebUI Script that enhances prompts using an LLM before generation.

    Lifecycle methods called by AUTOMATIC1111:

    * :py:meth:`title` — display name in the UI.
    * :py:meth:`show` — visibility control; always shown in both tabs.
    * :py:meth:`ui` — builds and returns the Gradio component list.
    * :py:meth:`run` — called just before image generation; injects the LLM
      response into the processing object's ``prompt`` attribute.
    """

    def __init__(self) -> None:
        """Initialise shared per-instance state."""
        super().__init__()
        self._last_response: Optional[ReasoningResponse] = None
        self._history: HistoryManager = HistoryManager(max_size=50)
        self._cached_config: Optional[dict] = None

    # ------------------------------------------------------------------
    # SD WebUI lifecycle — required overrides
    # ------------------------------------------------------------------

    def title(self) -> str:
        """Return the human-readable name shown in the SD WebUI accordion.

        Returns
        -------
        str
            Display title for this script.
        """
        return "Auto Prompt Reason"

    def show(self, is_img2img: bool) -> Any:
        """Control in which generation tabs this script is visible.

        Returning ``scripts.AlwaysVisible`` means the extension appears in
        **both** the txt2img and img2img tabs.

        Parameters
        ----------
        is_img2img:
            ``True`` when the script is being rendered in the img2img tab.

        Returns
        -------
        scripts.AlwaysVisible
            Sentinel value understood by SD WebUI.
        """
        return scripts.AlwaysVisible  # type: ignore[union-attr]

    def ui(self, is_img2img: bool) -> list:
        """Build and return the Gradio UI components for this script.

        The list order here **must** match the positional ``*args`` order
        in :py:meth:`run`.

        Parameters
        ----------
        is_img2img:
            ``True`` when building UI for the img2img tab.

        Returns
        -------
        list
            Ordered list of Gradio components:
            ``[enabled, provider_type, model_name, base_url, api_key,
            reasoning_effort, user_prompt, thinking_textbox,
            total_tokens_box, gen_time_box, provider_model_box]``
        """
        _gr = gr  # capture module reference; avoids repeated None checks
        with _gr.Group():  # type: ignore[union-attr]
            with _gr.Accordion("Auto Prompt Reason", open=False):  # type: ignore[union-attr]
                enabled = _gr.Checkbox(  # type: ignore[union-attr]
                    value=False,
                    label="Enable Auto Prompt Reason",
                )
                provider_type = _gr.Dropdown(  # type: ignore[union-attr]
                    choices=["ollama", "openai_compatible", "gemini"],
                    value="ollama",
                    label="Provider",
                )
                model_name = _gr.Textbox(  # type: ignore[union-attr]
                    value="",
                    label="Model Name",
                )
                base_url = _gr.Textbox(  # type: ignore[union-attr]
                    value="http://localhost:11434",
                    label="Base URL",
                )
                api_key = _gr.Textbox(  # type: ignore[union-attr]
                    value="",
                    label="API Key",
                    type="password",
                )
                reasoning_effort = _gr.Radio(  # type: ignore[union-attr]
                    choices=["low", "medium", "high"],
                    value="medium",
                    label="Reasoning Effort",
                )
                user_prompt = _gr.Textbox(  # type: ignore[union-attr]
                    value="",
                    label="Prompt for LLM",
                    lines=3,
                    placeholder="Describe the image you want to generate...",
                )

                # Display components — populated by postprocess() after generation
                _accordion, thinking_textbox = build_thinking_display(visible=True)
                total_tokens_box, gen_time_box, provider_model_box = build_metrics_display()

        return [
            enabled,
            provider_type,
            model_name,
            base_url,
            api_key,
            reasoning_effort,
            user_prompt,
            thinking_textbox,
            total_tokens_box,
            gen_time_box,
            provider_model_box,
        ]

    def run(
        self,
        p: Any,
        enabled: bool,
        provider_type: str,
        model_name: str,
        base_url: str,
        api_key: str,
        reasoning_effort: str,
        user_prompt: str,
        *_: Any,
    ) -> None:
        """Enhance the prompt via LLM and inject the result into *p*.

        Called by SD WebUI just before image generation begins.  The method is
        invoked on SD WebUI's worker thread so blocking HTTP calls are safe
        here.

        Parameters
        ----------
        p:
            SD WebUI processing object.  This method reads and writes
            ``p.prompt`` only.
        enabled:
            Whether the extension is active for this generation.
        provider_type:
            One of ``"ollama"``, ``"openai_compatible"``, or ``"gemini"``.
        model_name:
            Model identifier string (e.g. ``"llama3.2:latest"``).
        base_url:
            Root URL of the provider API endpoint.
        api_key:
            Optional API credential.  Empty string means no key is used.
        reasoning_effort:
            Reasoning depth hint — ``"low"``, ``"medium"``, or ``"high"``.
        user_prompt:
            The text prompt sent to the LLM.
        *_:
            Absorbs display-only args from ui() (thinking_textbox,
            total_tokens_box, gen_time_box, provider_model_box); ignored by
            run() — updated in postprocess().

        Returns
        -------
        None
            SD WebUI continues its normal processing pipeline regardless of
            the return value.
        """
        if not enabled:
            return None

        try:
            # Load config before making the LLM call so injection mode is
            # available immediately after the response is received.
            config = self._load_config()
            injection_mode: str = config.get("prompt_injection", "append")

            # Build provider kwargs; only pass api_key when non-empty.
            kwargs: dict[str, Any] = {
                "base_url": base_url,
                "model_name": model_name,
            }
            if api_key:
                kwargs["api_key"] = SecretStr(api_key)

            client = LLMClient.create(provider_type, **kwargs)

            _log.debug(
                "AutoPromptReason.run: calling provider=%r model=%r effort=%r",
                provider_type,
                model_name,
                reasoning_effort,
            )

            response: ReasoningResponse = client.generate(
                user_prompt,
                reasoning_effort=reasoning_effort,
            )

            # Persist response for potential use by postprocess().
            self._last_response = response

            # Record in session history.
            self._history.add_from_response(
                original_prompt=user_prompt,
                response=response,
                reasoning_effort=reasoning_effort,
            )

            # Inject the LLM answer into the processing prompt.
            if injection_mode == "replace":
                p.prompt = response.final_answer
            elif injection_mode == "prepend":
                p.prompt = f"{response.final_answer}, {p.prompt}"
            else:
                # Default: append
                p.prompt = f"{p.prompt}, {response.final_answer}"

            _log.info(
                "AutoPromptReason: prompt injected (mode=%r, provider=%r)",
                injection_mode,
                provider_type,
            )

        except Exception:  # noqa: BLE001
            _log.exception(
                "AutoPromptReason.run: error during LLM generation — "
                "prompt unchanged."
            )

        return None

    def postprocess(self, p: Any, processed: Any, *args: Any) -> None:
        """Post-processing hook: updates display components with last LLM response.

        Called by SD WebUI after image generation. Uses ``gr.update()`` to
        populate the thinking and metrics textboxes with values from
        ``self._last_response``.

        Parameters
        ----------
        p:
            SD WebUI processing object.
        processed:
            The :class:`Processed` result object produced by SD WebUI.
        *args:
            Forwarded Gradio component values (same order as :py:meth:`ui`).
        """
        if gr is None or self._last_response is None:
            return

        resp = self._last_response
        (
            thinking_text,
            thinking_visible,
            total_tokens_str,
            gen_time_str,
            provider_model_str,
            _result_text,
        ) = update_thinking_display({
            "thinking_content": resp.thinking_content,
            "final_answer": resp.final_answer,
            "total_tokens": resp.total_tokens,
            "reasoning_time_ms": resp.reasoning_time_ms,
            "model": resp.model,
            "provider": resp.provider,
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load ``config.yaml`` from the extension root directory.

        Returns a dict with at least the ``prompt_injection`` key.  Falls back
        to safe defaults if the file is missing or cannot be parsed.

        Returns
        -------
        dict
            Configuration mapping.  Keys relevant to this class:

            ``prompt_injection``
                One of ``"append"`` (default), ``"prepend"``, or
                ``"replace"``.
        """
        if self._cached_config is not None:
            return self._cached_config
        defaults: dict[str, Any] = {"prompt_injection": "append"}
        config_path = _EXTENSION_DIR / "config.yaml"
        if not config_path.exists():
            self._cached_config = defaults
            return self._cached_config
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                raw: Any = yaml.safe_load(fh)
            if not isinstance(raw, dict):
                self._cached_config = defaults
                return self._cached_config
            # The config stores the mode under ui.prompt_injection_mode.
            ui_section = raw.get("ui", {})
            if isinstance(ui_section, dict):
                mode = ui_section.get("prompt_injection_mode", "append")
                defaults["prompt_injection"] = mode
            self._cached_config = defaults
            return self._cached_config
        except Exception:  # noqa: BLE001
            _log.warning(
                "AutoPromptReason._load_config: failed to load %s — "
                "using defaults.",
                config_path,
            )
            self._cached_config = defaults
            return self._cached_config
