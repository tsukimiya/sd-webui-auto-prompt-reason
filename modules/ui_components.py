"""
UI components module for sd-webui-auto-prompt-reason.
Provides Gradio 3.x component builders for displaying LLM reasoning and results.
"""
from __future__ import annotations

from typing import Any

try:
    import gradio as gr  # type: ignore
except ImportError:
    gr = None  # type: ignore[assignment]


def build_thinking_display(visible: bool = False) -> tuple[Any, Any]:
    """
    Creates the collapsible "Thinking" section using Gradio 3.x components.

    Args:
        visible (bool): Controls the initial visibility of the Accordion wrapper.

    Returns:
        tuple: (accordion, thinking_textbox) component objects.
    """
    if gr is None:
        return None, None

    with gr.Accordion("🧠 Model Thinking", open=False, visible=visible) as accordion:
        thinking_textbox = gr.Textbox(
            label="Thinking Content",
            lines=8,
            interactive=False
        )

    return accordion, thinking_textbox


def build_metrics_display() -> tuple[Any, Any, Any]:
    """
    Creates a row of metrics for tokens, timing, and provider/model.

    Returns:
        tuple: (total_tokens_box, gen_time_box, provider_model_box) Gradio Textbox components.
    """
    if gr is None:
        return None, None, None

    with gr.Row():
        total_tokens_box = gr.Textbox(label="Total Tokens", interactive=False)
        gen_time_box = gr.Textbox(label="Generation Time (ms)", interactive=False)
        provider_model_box = gr.Textbox(label="Provider / Model", interactive=False)

    return total_tokens_box, gen_time_box, provider_model_box


def build_result_display() -> tuple[Any, Any]:
    """
    Creates the section showing the generated prompt result.

    Returns:
        tuple: (result_textbox, copy_button) Gradio components.
    """
    if gr is None:
        return None, None

    with gr.Group():
        result_textbox = gr.Textbox(label="Generated Prompt", lines=4, interactive=False)
        copy_button = gr.Button("📋 Copy to Prompt", variant="secondary")

    return result_textbox, copy_button


def build_history_table() -> Any:
    """
    Creates a history table to display past generations.

    Returns:
        gr.Dataframe: The Gradio Dataframe component, or None if Gradio is unavailable.
    """
    if gr is None:
        return None

    history_table = gr.Dataframe(
        headers=["Time", "Provider", "Model", "Effort", "Tokens", "Generated Prompt"],
        interactive=False,
        wrap=True
    )

    return history_table


def update_thinking_display(response_data: dict) -> tuple:
    """
    Transforms response data into values to update Gradio components.
    A pure function (no Gradio dependency - just data transformation).

    Args:
        response_data (dict): Dictionary with keys:
            thinking_content, final_answer, total_tokens,
            reasoning_time_ms, model, provider.

    Returns:
        tuple: (thinking_text, thinking_visible, total_tokens_str,
                gen_time_str, provider_model_str, result_text)
    """
    thinking_content = response_data.get("thinking_content")
    thinking_visible = thinking_content is not None
    thinking_text = thinking_content if thinking_content is not None else ""

    total_tokens_str = str(response_data.get("total_tokens", 0))
    gen_time_str = f"{response_data.get('reasoning_time_ms', 0)} ms"
    provider_model_str = f"{response_data.get('provider', '')} / {response_data.get('model', '')}"
    result_text = response_data.get("final_answer", "")

    return (
        thinking_text,
        thinking_visible,
        total_tokens_str,
        gen_time_str,
        provider_model_str,
        result_text
    )
