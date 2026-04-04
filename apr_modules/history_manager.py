"""In-memory, thread-safe prompt history manager for sd-webui-auto-prompt-reason.

Maintains a bounded deque of HistoryEntry objects representing past prompt
generation requests. No disk persistence — all history is lost on extension
reload. Thread-safe for use from SD WebUI background threads.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from itertools import islice
from dataclasses import dataclass
from typing import Optional

from apr_modules.models.reasoning_response import ReasoningResponse


@dataclass
class HistoryEntry:
    """A single recorded prompt generation event.

    Attributes:
        timestamp: Unix timestamp (time.time()) at the moment the entry was created.
        original_prompt: The raw prompt text submitted by the user before enhancement.
        final_answer: The enhanced prompt returned by the LLM.
        thinking_content: The model's reasoning/thinking text, or ``None`` if the
            model did not produce any visible chain-of-thought output.
        model: The name of the model that processed the request.
        provider: The provider identifier (e.g. ``"ollama"``, ``"openai_compatible"``,
            ``"gemini"``).
        reasoning_effort: The effort level used for reasoning — one of ``"low"``,
            ``"medium"``, or ``"high"``.
        total_tokens: Total number of tokens consumed by the request (prompt +
            completion).
        reasoning_time_ms: Wall-clock time taken for the LLM call, in milliseconds.
    """

    timestamp: float
    original_prompt: str
    final_answer: str
    thinking_content: Optional[str]
    model: str
    provider: str
    reasoning_effort: str
    total_tokens: int
    reasoning_time_ms: int


class HistoryManager:
    """Thread-safe, in-memory store for prompt generation history.

    Keeps the most recent *max_size* :class:`HistoryEntry` objects in a
    :class:`collections.deque`.  When the deque is full, the oldest entry is
    silently dropped on each new :meth:`add` call (standard ``deque(maxlen=N)``
    behaviour).

    All public methods acquire ``self._lock`` before touching ``self._entries``
    so the manager is safe to call from multiple SD WebUI background threads
    simultaneously.

    Args:
        max_size: Maximum number of history entries to retain.  Defaults to 50.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._max_size: int = max_size
        self._entries: deque[HistoryEntry] = deque(maxlen=max_size)
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(self, entry: HistoryEntry) -> None:
        """Append *entry* to the history deque in a thread-safe manner.

        If the deque has already reached ``max_size``, the oldest entry is
        automatically evicted by the underlying :class:`collections.deque`.

        Args:
            entry: The :class:`HistoryEntry` to store.
        """
        with self._lock:
            self._entries.append(entry)

    def add_from_response(
        self,
        original_prompt: str,
        response: ReasoningResponse,
        reasoning_effort: str = "medium",
    ) -> None:
        """Build a :class:`HistoryEntry` from a :class:`ReasoningResponse` and add it.

        Convenience wrapper that constructs a :class:`HistoryEntry` from the
        fields exposed by *response*, stamps it with the current time, and
        delegates to :meth:`add`.

        Args:
            original_prompt: The raw user prompt that was sent to the LLM.
            response: The :class:`ReasoningResponse` returned by the provider
                client.
            reasoning_effort: The effort level used for this request —
                ``"low"``, ``"medium"``, or ``"high"``.  Defaults to
                ``"medium"``.
        """
        entry = HistoryEntry(
            timestamp=time.time(),
            original_prompt=original_prompt,
            final_answer=response.final_answer,
            thinking_content=response.thinking_content,
            model=response.model,
            provider=response.provider,
            reasoning_effort=reasoning_effort,
            total_tokens=response.total_tokens,
            reasoning_time_ms=response.reasoning_time_ms,
        )
        self.add(entry)

    def clear(self) -> None:
        """Remove all entries from the history deque in a thread-safe manner."""
        with self._lock:
            self._entries.clear()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_all(self) -> list[HistoryEntry]:
        """Return a snapshot of all stored entries, oldest first.

        The returned list is a shallow copy of the current deque contents.
        Mutations to the list do not affect the internal state of this manager.

        Returns:
            A ``list`` of :class:`HistoryEntry` objects ordered from oldest
            to newest.
        """
        with self._lock:
            return list(self._entries)

    def get_latest(self, n: int = 10) -> list[HistoryEntry]:
        """Return up to *n* of the most recent entries.

        Entries are returned in chronological order (oldest first within the
        slice).  If fewer than *n* entries have been recorded, all available
        entries are returned.

        Args:
            n: Maximum number of recent entries to return.  Defaults to 10.

        Returns:
            A ``list`` of at most *n* :class:`HistoryEntry` objects.
        """
        with self._lock:
            if n >= len(self._entries):
                return list(self._entries)
            # islice on reversed yields at most n items without copying the whole deque
            return list(islice(reversed(self._entries), n))[::-1]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of stored entries (thread-safe).

        Returns:
            Integer count of :class:`HistoryEntry` objects currently held.
        """
        with self._lock:
            return len(self._entries)
