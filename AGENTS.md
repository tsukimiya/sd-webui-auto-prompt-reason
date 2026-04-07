# AGENTS.md

Repository guidance for coding agents working in `sd-webui-auto-prompt-reason/`.

## Scope

- This is a Python extension for AUTOMATIC1111 Stable Diffusion WebUI.
- Treat `sd-webui-auto-prompt-reason/` as the effective repo root.
- Main code: `apr_modules/`
- SD WebUI entry point: `scripts/auto_prompt_reason.py`
- Tests: `tests/`
- Runtime defaults: `config.yaml`
- All questions, status updates, explanations, and final responses to the user must be written in Japanese.

## Existing agent rule files

- No `.cursorrules` file was found.
- No `.cursor/rules/` directory was found.
- No `.github/copilot-instructions.md` file was found.
- No existing repository-root `AGENTS.md` was found before this one.

## Environment and tooling

- Python target: 3.10
- Type checker: `pyright`
- `pyrightconfig.json` uses `typeCheckingMode: "basic"`
- Dependencies are managed via `requirements.txt`
- No dedicated formatter config was found
- No dedicated lint config was found beyond type checking
- Preserve existing file-local style; avoid repo-wide reformatting

## Setup

Run commands from `sd-webui-auto-prompt-reason/`.

```bash
pip install -r requirements.txt
```

## Build, lint, and test

There is no formal build command in this repository.

### Type check

```bash
pyright
```

### Full test suite

```bash
pytest tests/ -v --cov=modules --cov=scripts --cov-report=term-missing
```

### Single test file

```bash
pytest tests/test_parser.py -v
```

### Single test function

```bash
pytest tests/test_parser.py::TestParseThinkTags::test_both_tags_present -v
```

### Single test with coverage

```bash
pytest tests/test_parser.py::TestParseThinkTags::test_both_tags_present -v --cov=modules --cov=scripts
```

## Pytest notes

From `pyproject.toml`:

- `testpaths = ["tests"]`
- default addopts: `--tb=short -v`
- coverage fail-under: `80`
- coverage sources: `modules` and `scripts`

Note: the actual source directory is `apr_modules/`, but coverage config and README examples still reference `modules`. Do not “clean this up” unless the task is specifically about fixing coverage/config consistency.

## Architecture guidelines

- Keep provider-specific HTTP and request logic in `apr_modules/providers/`.
- Keep parsing/normalization logic in pure helpers such as `apr_modules/reasoning_parser.py`.
- Keep simple structured data in `apr_modules/models/`.
- Preserve the provider/factory split described in `apr_modules/providers/base_provider.py`.
- Do not mix UI, HTTP, parsing, and persistence responsibilities in one module.

## Imports

- In most modules, start with `from __future__ import annotations`.
- Group imports as: future, stdlib, third-party, local.
- Prefer absolute local imports like `from apr_modules...`.
- Keep a blank line between import groups.

## Formatting and layout

- Use 4-space indentation.
- Preserve the existing multi-line formatting style for long signatures and literals.
- Keep module docstrings at the top of files.
- Longer files often use divider comments like `# ---------------------------------------------------------------------------`.
- Avoid unrelated formatting churn.

## Docstrings and comments

- Public modules, classes, and many public methods use descriptive docstrings.
- NumPy-style sections (`Parameters`, `Returns`, `Raises`, `Examples`) are common.
- Some files use `Args:` / `Returns:` style; match the surrounding file instead of rewriting everything.
- Prefer comments that explain intent or constraints.
- Avoid unnecessary suppression comments; existing code uses targeted `# noqa: D401` only where needed.

## Types

- Add type annotations for public functions, methods, attributes, and fixtures.
- Match the local file’s style for `Optional[T]` and built-in generics like `dict[str, Any]`.
- Use explicit return types.
- Use `Any` only for genuinely dynamic structures such as raw API payloads.
- Avoid broad `# type: ignore`; if necessary, make it specific.

## Naming

- Classes: `CamelCase`
- Functions and methods: `snake_case`
- Variables and attributes: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers/state: leading underscore
- Test classes: `Test...`
- Test functions: `test_...`

## Data and validation patterns

- Use `@dataclass` for small structured models.
- Use `__post_init__` for lightweight validation.
- Use `@property` for derived values when it improves the model API.
- Keep validation focused and descriptive.

## Error handling

- Raise descriptive `ValueError` or `TypeError` for invalid input/configuration.
- Propagate real HTTP failures instead of hiding them.
- Graceful fallback is acceptable only where the existing code already does it.
- Do not add empty `except` blocks.

## Secrets and security

- Follow the existing `SecretStr` pattern.
- Do not log raw API keys.
- Only unwrap secrets where the HTTP request actually needs them.

## Concurrency

- This repo uses thread-based async behavior, not `asyncio`.
- Follow the existing `threading.Thread` + callback style in the client layer.
- Protect shared mutable state with `threading.Lock` when concurrency is involved.

## Testing conventions

- Use `pytest` and keep shared fixtures in `tests/conftest.py`.
- Prefer class-based test grouping.
- Mock HTTP and provider interactions; do not add live network tests.
- Add focused unit tests near the behavior you change.
- For parser changes, cover multiple response shapes.
- For client/provider changes, cover both success and failure paths.

## Change strategy

- Make minimal, local changes.
- Prefer extending the existing provider/parser/model structure over adding new abstractions.
- Do not rename packages or reorganize modules unless explicitly asked.
- Do not clean up unrelated code while fixing a bug.
- Keep README, tests, and config examples in sync when public behavior changes.

## Validation before finishing

- Run `pyright` for Python changes with type implications.
- Run the smallest relevant pytest command first.
- Run the full suite for broader behavior changes.
- Call out pre-existing issues separately from issues introduced by your change.
