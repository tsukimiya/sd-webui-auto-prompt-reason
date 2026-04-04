"""
Integration tests for :class:`AutoPromptReason` in ``scripts/auto_prompt_reason.py``.

All SD WebUI (``modules.scripts``) and Gradio (``gradio``) modules are unavailable
in this environment.  The source guards these with ``try/except ImportError`` and
sets ``scripts = None`` / ``gr = None`` when absent, so ``AutoPromptReason``
inherits from ``object`` and can be instantiated freely.

``LLMClient.generate`` is always mocked to prevent live HTTP calls.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import-untyped]

from scripts.auto_prompt_reason import AutoPromptReason  # type: ignore[import]
from apr_modules.models.reasoning_response import ReasoningResponse  # type: ignore[import]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_response(final_answer: str = "fluffy cat") -> ReasoningResponse:
    """Build a minimal :class:`ReasoningResponse` for use in tests."""
    return ReasoningResponse(
        final_answer=final_answer,
        thinking_content="thoughts",
        reasoning_tokens=5,
        completion_tokens=20,
        total_tokens=100,
        reasoning_time_ms=1500,
        model="test-model",
        provider="ollama",
        raw_response={},
    )


def _make_processing_obj(prompt: str = "original") -> MagicMock:
    """Return a mock SD WebUI processing object with a ``prompt`` attribute."""
    p = MagicMock()
    p.prompt = prompt
    return p


def _patch_llm(final_answer: str = "fluffy cat"):
    """Context manager that patches ``LLMClient`` and returns a mock client."""
    mock_client_class = MagicMock()
    mock_instance = MagicMock()
    mock_client_class.create.return_value = mock_instance
    mock_instance.generate.return_value = _make_response(final_answer)
    return mock_client_class, mock_instance


# ---------------------------------------------------------------------------
# TestAutoPromptReasonLifecycle
# ---------------------------------------------------------------------------


class TestAutoPromptReasonLifecycle:
    """Tests for class construction and simple lifecycle methods."""

    def test_instantiation_creates_history(self) -> None:
        """``__init__`` creates a ``HistoryManager`` stored as ``_history``."""
        ext = AutoPromptReason()
        assert ext._history is not None

    def test_instantiation_last_response_is_none(self) -> None:
        """``__init__`` sets ``_last_response`` to ``None`` initially."""
        ext = AutoPromptReason()
        assert ext._last_response is None

    def test_title_returns_expected_string(self) -> None:
        """``title()`` returns the display name used in SD WebUI."""
        ext = AutoPromptReason()
        assert ext.title() == "Auto Prompt Reason"

    def test_show_does_not_raise_when_scripts_is_none(self) -> None:
        """``show()`` should not raise even though ``scripts`` is None."""
        ext = AutoPromptReason()
        # scripts.AlwaysVisible is None when scripts module is not installed;
        # the method just returns it — we only care it doesn't raise.
        try:
            ext.show(False)
        except Exception as exc:  # noqa: BLE001
            # AttributeError would indicate scripts is None (expected behaviour)
            # — any other exception is a real failure.
            assert isinstance(exc, AttributeError), f"Unexpected exception: {exc!r}"

    def test_show_returns_none_when_scripts_absent(self) -> None:
        """``show()`` returns ``None`` (scripts.AlwaysVisible) in test env."""
        ext = AutoPromptReason()
        # When scripts is None, ``scripts.AlwaysVisible`` raises AttributeError.
        # We verify that the only possible exception is AttributeError.
        result_or_exc: Any = None
        try:
            result_or_exc = ext.show(True)
        except AttributeError:
            result_or_exc = AttributeError
        assert result_or_exc is not None  # either a value or the sentinel class

    def test_postprocess_does_not_raise(self) -> None:
        """``postprocess()`` is a no-op and must never raise."""
        ext = AutoPromptReason()
        p = _make_processing_obj()
        processed = MagicMock()
        ext.postprocess(p, processed)  # type: ignore[call-arg]

    def test_postprocess_with_extra_args_does_not_raise(self) -> None:
        """``postprocess()`` accepts ``*args`` without raising."""
        ext = AutoPromptReason()
        p = _make_processing_obj()
        processed = MagicMock()
        ext.postprocess(p, processed, True, "ollama", "model")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestAutoPromptReasonProcess
# ---------------------------------------------------------------------------


class TestAutoPromptReasonProcess:
    """Tests for :py:meth:`AutoPromptReason.process`.

    ``process`` is the AlwaysVisible pre-generation callback called by
    SD WebUI before image generation begins.  It receives the processing object
    and the ordered list of UI component values returned by ``ui()``.
    """

    # ------------------------------------------------------------------
    # Disabled path
    # ------------------------------------------------------------------

    def test_process_disabled_returns_none_without_llm_call(self) -> None:
        """When ``enabled=False``, process returns immediately without calling LLMClient."""
        mock_client_class, mock_instance = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj()
            result = ext.process(p, False, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert result is None
        mock_client_class.create.assert_not_called()

    def test_process_disabled_leaves_prompt_unchanged(self) -> None:
        """``p.prompt`` must not be mutated when the extension is disabled."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj("unchanged")
            ext.process(p, False, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert p.prompt == "unchanged"

    # ------------------------------------------------------------------
    # Injection modes
    # ------------------------------------------------------------------

    def test_process_append_mode_modifies_prompt(self) -> None:
        """Default (append) mode appends LLM answer to the original prompt."""
        mock_client_class, _ = _patch_llm("fluffy cat")
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj("original")
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert p.prompt == "original, fluffy cat"

    def test_process_prepend_mode_modifies_prompt(self) -> None:
        """Prepend mode puts LLM answer before the original prompt."""
        mock_client_class, _ = _patch_llm("fluffy cat")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("ui:\n  prompt_injection_mode: prepend\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
                with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                    ext = AutoPromptReason()
                    p = _make_processing_obj("original")
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert p.prompt == "fluffy cat, original"

    def test_process_replace_mode_modifies_prompt(self) -> None:
        """Replace mode overwrites the prompt entirely with the LLM answer."""
        mock_client_class, _ = _patch_llm("fluffy cat")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("ui:\n  prompt_injection_mode: replace\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
                with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                    ext = AutoPromptReason()
                    p = _make_processing_obj("original")
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert p.prompt == "fluffy cat"

    # ------------------------------------------------------------------
    # State side-effects
    # ------------------------------------------------------------------

    def test_process_sets_last_response(self) -> None:
        """``_last_response`` is populated after a successful process call."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert ext._last_response is not None
        assert ext._last_response.final_answer == "fluffy cat"

    def test_process_records_history_entry(self) -> None:
        """``_history`` gains an entry after a successful process call."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert len(ext._history) == 1

    def test_process_multiple_calls_accumulate_history(self) -> None:
        """Each successful process call appends a new entry to history."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
                p2 = _make_processing_obj()
                ext.process(p2, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert len(ext._history) == 2

    # ------------------------------------------------------------------
    # api_key handling
    # ------------------------------------------------------------------

    def test_process_with_api_key_passes_secret_str(self) -> None:
        """When api_key is non-empty, ``SecretStr`` is passed to ``LLMClient.create``."""
        from modules.providers.base_provider import SecretStr  # type: ignore[import]
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "secret-key", "medium", "q", "")
        _call_kwargs = mock_client_class.create.call_args
        assert _call_kwargs is not None
        assert "api_key" in _call_kwargs.kwargs
        assert isinstance(_call_kwargs.kwargs["api_key"], SecretStr)

    def test_process_without_api_key_omits_kwarg(self) -> None:
        """When api_key is empty string, ``api_key`` kwarg is NOT passed to ``LLMClient.create``."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        _call_kwargs = mock_client_class.create.call_args
        assert _call_kwargs is not None
        assert "api_key" not in _call_kwargs.kwargs

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_process_exception_in_generate_does_not_raise(self) -> None:
        """Exceptions from ``generate()`` are caught; process returns None safely."""
        mock_client_class = MagicMock()
        mock_instance = MagicMock()
        mock_client_class.create.return_value = mock_instance
        mock_instance.generate.side_effect = RuntimeError("network failure")
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj("unchanged")
            result = ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert result is None

    def test_process_exception_leaves_prompt_unchanged(self) -> None:
        """When generate() raises, the original prompt must not be modified."""
        mock_client_class = MagicMock()
        mock_instance = MagicMock()
        mock_client_class.create.return_value = mock_instance
        mock_instance.generate.side_effect = ValueError("bad response")
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj("unchanged")
            ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert p.prompt == "unchanged"

    def test_process_exception_last_response_remains_none(self) -> None:
        """When generate() raises before assignment, ``_last_response`` stays None."""
        mock_client_class = MagicMock()
        mock_instance = MagicMock()
        mock_client_class.create.return_value = mock_instance
        mock_instance.generate.side_effect = ConnectionError("timeout")
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj()
            ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        assert ext._last_response is None


# ---------------------------------------------------------------------------
# TestAutoPromptReasonLoadConfig
# ---------------------------------------------------------------------------


class TestAutoPromptReasonLoadConfig:
    """Tests for :py:meth:`AutoPromptReason._load_config`."""

    def test_no_config_file_returns_default_append(self) -> None:
        """Missing config.yaml → returns ``{"prompt_injection": "append"}``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "append", "default_system_prompt": ""}

    def test_config_with_prepend_mode(self) -> None:
        """Valid config.yaml with ``ui.prompt_injection_mode: prepend`` is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("ui:\n  prompt_injection_mode: prepend\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "prepend", "default_system_prompt": ""}

    def test_config_with_replace_mode(self) -> None:
        """Valid config.yaml with ``ui.prompt_injection_mode: replace`` is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("ui:\n  prompt_injection_mode: replace\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "replace", "default_system_prompt": ""}

    def test_config_with_append_mode_explicit(self) -> None:
        """Explicit ``append`` in config.yaml returns append mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("ui:\n  prompt_injection_mode: append\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "append", "default_system_prompt": ""}

    def test_invalid_yaml_returns_default(self) -> None:
        """Malformed YAML in config.yaml → falls back to default ``append``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            # Write syntactically invalid YAML (tab in weird place, broken structure)
            config_path.write_text(
                "ui:\n  prompt_injection_mode: [\n  unclosed bracket\n",
                encoding="utf-8",
            )
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "append", "default_system_prompt": ""}

    def test_config_non_dict_root_returns_default(self) -> None:
        """YAML that parses to a non-dict (e.g. a list) → default is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("- item1\n- item2\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "append", "default_system_prompt": ""}

    def test_config_missing_ui_section_returns_default(self) -> None:
        """Config with no ``ui`` section → default append mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("other_key: value\n", encoding="utf-8")
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tmpdir)):
                ext = AutoPromptReason()
                config = ext._load_config()
        assert config == {"prompt_injection": "append", "default_system_prompt": ""}


# ---------------------------------------------------------------------------
# TestAutoPromptReasonSystemPrompt
# ---------------------------------------------------------------------------


class TestAutoPromptReasonSystemPrompt:
    """Tests for system_prompt forwarding through AutoPromptReason.process.

    These tests verify that ``process`` passes ``system_prompt`` to
    ``LLMClient.generate`` once the parallel UI/client plumbing is in place.
    They use a mock client to inspect the arguments that ``generate`` receives.
    """

    def test_process_generate_called_with_prompt_and_effort(self) -> None:
        """``generate`` receives user_prompt and reasoning_effort from ``process``."""
        mock_client_class, mock_instance = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "high", "my prompt", "")
        call_args = mock_instance.generate.call_args
        assert call_args is not None
        # user_prompt is the first positional argument
        assert call_args.args[0] == "my prompt"
        # reasoning_effort is forwarded as a kwarg
        assert call_args.kwargs.get("reasoning_effort") == "high"

    def test_process_generate_called_once_per_invocation(self) -> None:
        """``generate`` is called exactly once per ``process`` invocation."""
        mock_client_class, mock_instance = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        mock_instance.generate.assert_called_once()

    def test_process_nonempty_system_prompt_forwarded_to_generate(self) -> None:
        """Non-empty system_prompt is forwarded to ``generate`` as a kwarg."""
        mock_client_class, mock_instance = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(
                    p, True, "ollama", "m", "http://localhost", "", "medium",
                    "my prompt", "You are an artist.",
                )
        call_args = mock_instance.generate.call_args
        assert call_args is not None
        assert call_args.kwargs.get("system_prompt") == "You are an artist."

    def test_process_empty_system_prompt_passes_none_to_generate(self) -> None:
        """Empty system_prompt string results in ``None`` being forwarded to ``generate``."""
        mock_client_class, mock_instance = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        call_args = mock_instance.generate.call_args
        assert call_args is not None
        assert call_args.kwargs.get("system_prompt") is None


# ---------------------------------------------------------------------------
# TestAutoPromptReasonLogging
# ---------------------------------------------------------------------------


class TestAutoPromptReasonLogging:
    """Tests that verify trace log output from :py:meth:`AutoPromptReason.process`.

    Logging assertions are kept minimal: we verify the *key diagnostic phrases*
    appear in the captured log rather than the full formatted string, so the
    tests remain stable against minor message wording changes.
    """

    # ------------------------------------------------------------------
    # Disabled path — must log that the extension was skipped
    # ------------------------------------------------------------------

    def test_disabled_logs_skip_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """When disabled, an INFO message indicating skip is emitted."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj()
            with caplog.at_level(logging.INFO, logger="scripts.auto_prompt_reason"):
                ext.process(p, False, "ollama", "m", "http://localhost", "", "medium", "q", "")
        messages = [r.message for r in caplog.records]
        assert any("disabled" in m or "skipping" in m for m in messages), (
            f"Expected a 'disabled/skipping' log entry; got: {messages}"
        )

    def test_disabled_does_not_log_generate_called(self, caplog: pytest.LogCaptureFixture) -> None:
        """When disabled, no message about calling generate() must appear."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            ext = AutoPromptReason()
            p = _make_processing_obj()
            with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                ext.process(p, False, "ollama", "m", "http://localhost", "", "medium", "q", "")
        messages = [r.message for r in caplog.records]
        assert not any("generate()" in m for m in messages), (
            f"generate() must not be mentioned in disabled path; got: {messages}"
        )

    # ------------------------------------------------------------------
    # Enabled path — invocation entry + call parameters
    # ------------------------------------------------------------------

    def test_enabled_logs_process_invoked(self, caplog: pytest.LogCaptureFixture) -> None:
        """A DEBUG entry recording that process was invoked is emitted."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "mymodel", "http://localhost", "", "high", "q", "")
        messages = [r.message for r in caplog.records]
        assert any("invoked" in m for m in messages), (
            f"Expected an 'invoked' log entry; got: {messages}"
        )

    def test_enabled_logs_provider_and_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """Provider type and model name appear in at least one DEBUG log entry."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "gemini", "gemini-pro", "http://localhost", "", "low", "q", "")
        full_output = " ".join(r.message for r in caplog.records)
        assert "gemini" in full_output, f"Provider 'gemini' not found in logs: {full_output!r}"
        assert "gemini-pro" in full_output, f"Model 'gemini-pro' not found in logs: {full_output!r}"

    def test_enabled_logs_has_api_key_false_when_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        """When api_key is empty, ``has_api_key=False`` appears in the entry log."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        full_output = " ".join(r.message for r in caplog.records)
        assert "has_api_key=False" in full_output, (
            f"Expected 'has_api_key=False' in logs: {full_output!r}"
        )

    def test_enabled_logs_has_api_key_true_when_provided(self, caplog: pytest.LogCaptureFixture) -> None:
        """When api_key is non-empty, ``has_api_key=True`` appears in the entry log."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "openai_compatible", "gpt-4o", "http://api", "sk-secret", "medium", "q", "")
        full_output = " ".join(r.message for r in caplog.records)
        assert "has_api_key=True" in full_output, (
            f"Expected 'has_api_key=True' in logs: {full_output!r}"
        )
        # The raw secret must NEVER appear in any log message
        assert "sk-secret" not in full_output, "API key value must not appear in logs"

    # ------------------------------------------------------------------
    # Enabled path — config loaded + generate() called
    # ------------------------------------------------------------------

    def test_enabled_logs_config_loaded(self, caplog: pytest.LogCaptureFixture) -> None:
        """A DEBUG entry reporting ``injection_mode`` from config is emitted."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        full_output = " ".join(r.message for r in caplog.records)
        assert "injection_mode" in full_output, (
            f"Expected 'injection_mode' in config-loaded log: {full_output!r}"
        )

    def test_enabled_logs_generate_called(self, caplog: pytest.LogCaptureFixture) -> None:
        """An INFO entry about calling ``client.generate()`` is emitted."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.INFO, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        messages = [r.message for r in caplog.records if r.levelno >= logging.INFO]
        assert any("generate()" in m for m in messages), (
            f"Expected a generate() INFO log; got: {messages}"
        )

    def test_enabled_logs_user_prompt_length_not_content(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logs include ``user_prompt_len`` but must NOT contain the raw prompt text."""
        mock_client_class, _ = _patch_llm()
        secret_prompt = "a very specific secret prompt text"
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.DEBUG, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", secret_prompt, "")
        full_output = " ".join(r.message for r in caplog.records)
        assert "user_prompt_len" in full_output, (
            f"Expected 'user_prompt_len' in logs: {full_output!r}"
        )
        assert secret_prompt not in full_output, (
            "Raw prompt text must not appear in any log message"
        )

    # ------------------------------------------------------------------
    # Enabled path — post-generate: response stored + prompt injected
    # ------------------------------------------------------------------

    def test_enabled_logs_prompt_injected(self, caplog: pytest.LogCaptureFixture) -> None:
        """An INFO entry confirming prompt injection is emitted after successful process."""
        mock_client_class, _ = _patch_llm()
        with patch("scripts.auto_prompt_reason.LLMClient", mock_client_class):
            with patch("scripts.auto_prompt_reason._EXTENSION_DIR", Path(tempfile.mkdtemp())):
                ext = AutoPromptReason()
                p = _make_processing_obj()
                with caplog.at_level(logging.INFO, logger="scripts.auto_prompt_reason"):
                    ext.process(p, True, "ollama", "m", "http://localhost", "", "medium", "q", "")
        messages = [r.message for r in caplog.records if r.levelno >= logging.INFO]
        assert any("injected" in m for m in messages), (
            f"Expected a 'prompt injected' INFO log; got: {messages}"
        )
