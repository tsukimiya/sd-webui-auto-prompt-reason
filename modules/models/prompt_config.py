"""Reasoning configuration data model."""

from dataclasses import dataclass


@dataclass
class ReasoningConfig:
    # Reasoning effort level: "low", "medium", or "high"
    reasoning_effort: str = "medium"
    # Whether to display thinking in UI
    show_thinking: bool = True
    # Seconds before timeout
    thinking_timeout: int = 60
    # Max tokens for thinking
    max_reasoning_tokens: int = 8192
    # Max tokens for final answer
    max_completion_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("reasoning_effort must be 'low', 'medium', or 'high'")

    @property
    def temperature_for_effort(self) -> float:
        if self.reasoning_effort == "low":
            return 0.8
        if self.reasoning_effort == "medium":
            return 0.5
        return 0.2
