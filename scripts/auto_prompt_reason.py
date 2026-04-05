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
* The ``process()`` method is called by SD WebUI on a worker thread — it is safe to
  call ``LLMClient.generate()`` synchronously here.
* No global mutable config state is used; config is loaded once per instance
  and cached via :py:meth:`AutoPromptReason._load_config`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

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

# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------
# SD WebUI's logging_config.setup_logging() only installs root handlers when
# --loglevel / SD_WEBUI_LOG_LEVEL is explicitly set.  When neither is provided
# the root logger has no handlers and every getLogger(...) call silently drops
# its messages.  Install a fallback StreamHandler so this extension's logs
# always reach the console regardless of the host's log configuration.
_log = logging.getLogger(__name__)
if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.root.addHandler(_handler)
    logging.root.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Extension root — used to locate config.yaml at runtime
# ---------------------------------------------------------------------------

_EXTENSION_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PROVIDER_TIMEOUT = (10, 180)


# ---------------------------------------------------------------------------
# Script class
# ---------------------------------------------------------------------------


class AutoPromptReason(scripts.Script):  # type: ignore[misc,valid-type]
    """SD WebUI Script that enhances prompts using an LLM before generation.

    Lifecycle methods called by AUTOMATIC1111:

    * :py:meth:`title` — display name in the UI.
    * :py:meth:`show` — visibility control; always shown in both tabs.
    * :py:meth:`ui` — builds and returns the Gradio component list.
    * :py:meth:`process` — called just before image generation; injects the LLM
      response into ``p.all_prompts`` for all images in the batch.
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
        in :py:meth:`process`.

        Parameters
        ----------
        is_img2img:
            ``True`` when building UI for the img2img tab.

        Returns
        -------
        list
            Ordered list of Gradio components:
            ``[enabled, provider_type, model_name, base_url, api_key,
            reasoning_effort, user_prompt, system_prompt, thinking_textbox,
            total_tokens_box, gen_time_box, provider_model_box]``
        """
        config = self._load_config()
        default_system_prompt: str = config.get("default_system_prompt", "")

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
                system_prompt = _gr.Textbox(  # type: ignore[union-attr]
                    value=default_system_prompt,
                    label="System Prompt",
                    lines=4,
                    placeholder=default_system_prompt,
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
            system_prompt,
            thinking_textbox,
            total_tokens_box,
            gen_time_box,
            provider_model_box,
        ]

    def process(
        self,
        p: Any,
        enabled: bool,
        provider_type: str,
        model_name: str,
        base_url: str,
        api_key: str,
        reasoning_effort: str,
        user_prompt: str,
        system_prompt: str,
        *_: Any,
    ) -> None:
        """Enhance the prompt via LLM and inject the result into *p*.

        Called by SD WebUI just before image generation begins on the worker
        thread, making it safe to call ``LLMClient.generate()`` synchronously.

        This is the correct always-on pre-generation callback for
        ``AlwaysVisible`` scripts; ``run()`` is not invoked for such scripts.

        Parameters
        ----------
        p:
            SD WebUI processing object.  This method writes to
            ``p.all_prompts`` (the authoritative per-batch prompt list used
            by the sampler).
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
        system_prompt:
            System-level instruction for the LLM.  Empty string is treated as
            ``None`` (no system message).
        *_:
            Absorbs display-only args from ui() (thinking_textbox,
            total_tokens_box, gen_time_box, provider_model_box); ignored by
            process() — updated in postprocess().

        Returns
        -------
        None
            SD WebUI continues its normal processing pipeline regardless of
            the return value.
        """
        _log.info(
            "AutoPromptReason.process: invoked enabled=%r provider=%r model=%r"
            " effort=%r base_url=%r has_api_key=%r"
            " user_prompt_len=%d has_system_prompt=%r",
            enabled,
            provider_type,
            model_name,
            reasoning_effort,
            base_url,
            bool(api_key),
            len(user_prompt),
            bool(system_prompt),
        )

        if not enabled:
            _log.info(
                "AutoPromptReason.process: extension disabled — skipping LLM call."
            )
            return None

        try:
            # Load config before making the LLM call so injection mode is
            # available immediately after the response is received.
            config = self._load_config()
            injection_mode: str = config.get("prompt_injection", "append")
            provider_timeout = cast(tuple[int, int], config.get("provider_timeout", _DEFAULT_PROVIDER_TIMEOUT))
            _log.info(
                "AutoPromptReason.process: config loaded injection_mode=%r provider_timeout=%r",
                injection_mode,
                provider_timeout,
            )

            # Build provider kwargs; only pass api_key when non-empty.
            kwargs: dict[str, Any] = {
                "base_url": base_url,
                "model_name": model_name,
                "timeout": provider_timeout,
            }
            if api_key:
                kwargs["api_key"] = SecretStr(api_key)

            _log.info(
                "AutoPromptReason.process: creating LLMClient"
                " provider=%r model=%r base_url=%r has_api_key=%r timeout=%r",
                provider_type,
                model_name,
                base_url,
                bool(api_key),
                provider_timeout,
            )
            client = LLMClient.create(provider_type, **kwargs)
            _log.info("AutoPromptReason.process: LLMClient.create() returned %r", type(client).__name__)

            # Treat empty string as "no system prompt" so the provider falls
            # back to its own defaults.
            effective_system_prompt: Optional[str] = system_prompt or None

            _log.info(
                "AutoPromptReason.process: calling client.generate()"
                " provider=%r model=%r effort=%r"
                " user_prompt_len=%d has_system_prompt=%r",
                provider_type,
                model_name,
                reasoning_effort,
                len(user_prompt),
                effective_system_prompt is not None,
            )
            response: ReasoningResponse = client.generate(
                user_prompt,
                reasoning_effort=reasoning_effort,
                system_prompt=effective_system_prompt,
            )
            _log.info(
                "AutoPromptReason.process: generate() returned"
                " final_answer=%.300r thinking_content=%s"
                " total_tokens=%r reasoning_time_ms=%r",
                response.final_answer,
                "present" if response.thinking_content else "None",
                response.total_tokens,
                response.reasoning_time_ms,
            )

            # final_answerが空文字列の場合は警告（原因特定のため raw_response の message 内容も出力）
            if not response.final_answer or not response.final_answer.strip():
                _raw = response.raw_response
                _msg_dump: Any = None
                if isinstance(_raw, dict):
                    _choices = _raw.get("choices", [])
                    if _choices and isinstance(_choices[0], dict):
                        _msg_dump = _choices[0].get("message")
                _log.warning(
                    "[apr][script][inject] WARNING: final_answer is EMPTY — "
                    "prompt will NOT be changed. raw_response_keys=%s"
                    " choices[0].message=%r",
                    sorted(_raw.keys()) if isinstance(_raw, dict) else type(_raw).__name__,
                    _msg_dump,
                )

            # Persist response for potential use by postprocess().
            self._last_response = response
            _log.debug("AutoPromptReason.process: _last_response stored")

            # Record in session history.
            self._history.add_from_response(
                original_prompt=user_prompt,
                response=response,
                reasoning_effort=reasoning_effort,
            )
            _log.debug(
                "AutoPromptReason.process: history written history_size=%d",
                len(self._history),
            )

            # Inject the LLM answer into p.all_prompts (the authoritative list
            # used by the sampler).  p.prompt is only used once by
            # setup_prompts() to build all_prompts before process() is called;
            # writing to p.prompt afterwards has no effect on the actual
            # generation.  Iterate over all entries to cover batch generation.
            # Reference: xlinx/sd-webui-decadetw-auto-prompt-llm uses the
            # same p.all_prompts loop pattern.
            answer = response.final_answer
            original_prompt_len = len(p.all_prompts[0]) if p.all_prompts else len(p.prompt)

            for i in range(len(p.all_prompts)):
                original = p.all_prompts[i]
                if injection_mode == "replace":
                    p.all_prompts[i] = answer
                elif injection_mode == "prepend":
                    sep = ", " if original else ""
                    p.all_prompts[i] = f"{answer}{sep}{original}"
                else:
                    # Default: append
                    sep = ", " if original else ""
                    p.all_prompts[i] = f"{original}{sep}{answer}"

            injected_sample = p.all_prompts[0] if p.all_prompts else ""
            _log.info(
                "AutoPromptReason.process: prompt injected"
                " mode=%r provider=%r all_prompts_count=%d"
                " original_len=%d result_len=%d"
                " injected_prompt=%.500r",
                injection_mode,
                provider_type,
                len(p.all_prompts),
                original_prompt_len,
                len(injected_sample),
                injected_sample,
            )

        except Exception:  # noqa: BLE001
            _log.exception(
                "AutoPromptReason.process: error during LLM generation"
                " provider=%r model=%r base_url=%r timeout=%r — "
                "prompt unchanged.",
                provider_type,
                model_name,
                base_url,
                provider_timeout if 'provider_timeout' in locals() else _DEFAULT_PROVIDER_TIMEOUT,
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

    def _load_config(self, provider_type: str = "ollama") -> dict:
        """Load ``config.yaml`` from the extension root directory.

        Returns a dict with at least the ``prompt_injection``,
        ``default_system_prompt``, and ``provider_timeout`` keys. Falls back to
        safe defaults if the
        file is missing or cannot be parsed.

        Returns
        -------
        dict
            Configuration mapping.  Keys relevant to this class:

            ``prompt_injection``
                One of ``"append"`` (default), ``"prepend"``, or
                ``"replace"``.
            ``default_system_prompt``
                System prompt string shown as the textbox default value.
                Empty string when not configured.
            ``provider_timeout``
                Two-element timeout tuple ``(connect_timeout_s, read_timeout_s)``.
        """
        if self._cached_config is not None:
            return self._cached_config
        defaults: dict[str, Any] = {
            "prompt_injection": "append",
            "default_system_prompt": "",
            "provider_timeout": _DEFAULT_PROVIDER_TIMEOUT,
        }
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
                default_sp = ui_section.get("default_system_prompt", "")
                defaults["default_system_prompt"] = default_sp if isinstance(default_sp, str) else ""
            provider_section = raw.get(provider_type, {})
            if isinstance(provider_section, dict):
                timeout_value = provider_section.get("timeout")
                if timeout_value is None:
                    # null / 未設定 → 安全なデフォルト (connect=10s, read=180s)
                    # read_timeout=None (無制限) はCloudflare等のリバースプロキシが
                    # 先にタイムアウトして524エラーを返す原因になるため避ける
                    defaults["provider_timeout"] = _DEFAULT_PROVIDER_TIMEOUT
                    _log.debug(
                        "AutoPromptReason._load_config: timeout is null — "
                        "using default timeout=%r",
                        _DEFAULT_PROVIDER_TIMEOUT,
                    )
                elif isinstance(timeout_value, (int, float)) and timeout_value > 0:
                    defaults["provider_timeout"] = (10, int(timeout_value))
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
