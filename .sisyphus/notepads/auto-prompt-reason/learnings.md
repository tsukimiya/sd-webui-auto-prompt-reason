# Learnings — auto-prompt-reason

## Provider base class (2026-04-04)

- Python name-mangling (`self.__secret_value` → `self._SecretStr__secret_value`) is
  correctly handled within the same class: `other.__secret_value` inside a
  `SecretStr` method also resolves to the mangled name, so `__eq__` works without
  any extra workaround. Pyright raises a false-positive `attr-defined` here;
  `# type: ignore[attr-defined]` suppresses it cleanly.

- `from __future__ import annotations` turns all annotations into strings, making
  forward refs like `-> "ReasoningResponse"` redundant at the syntax level but
  harmless and explicit as documentation. Pyright still warns about unresolved
  names in annotation strings, which is expected and intentional for circular-
  import avoidance.

- No Python 3 runtime on the dev machine (only Python 2.7 + Windows Store stub).
  LSP diagnostics via Pyright are the primary verification mechanism for syntax
  correctness. The only error reported is the intentional `ReasoningResponse`
  forward reference.

- `ProviderType` enum values are lowercase strings (`"ollama"`, etc.) matching
  the pattern used in config/settings JSON — avoids case-conversion ceremony
  downstream.
