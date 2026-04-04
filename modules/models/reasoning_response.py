"""Reasoning response data model."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ReasoningResponse:
    # The actual prompt/answer (stripped of <think> tags)
    final_answer: str
    # Tokens in final answer
    completion_tokens: int
    # Total tokens used
    total_tokens: int
    # Model name used
    model: str
    # Provider name (e.g., "ollama")
    provider: str
    # Original API response (for debugging)
    raw_response: dict[str, Any]
    # Reasoning/thinking text (None if not available)
    thinking_content: Optional[str] = None
    # Tokens used for thinking (None if N/A)
    reasoning_tokens: Optional[int] = None
    # Total generation time in milliseconds
    reasoning_time_ms: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens < self.completion_tokens:
            raise ValueError("total_tokens must be greater than or equal to completion_tokens")

    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking_content and self.thinking_content.strip())

    @property
    def thinking_ratio(self) -> float:
        if not self.has_thinking or not self.reasoning_tokens or self.total_tokens <= 0:
            return 0.0
        return self.reasoning_tokens / self.total_tokens
