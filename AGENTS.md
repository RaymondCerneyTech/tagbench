# AGENTS.md (Repo House Rules)

These rules apply to the entire repository.

## Prompts
- Keep prompts minimal and unambiguous.
- Prefer programmatic evaluation over subjective judging.

## Change size
- Prefer small, PR-sized changes (focused diffs; avoid drive-by refactors).
- If a large change is necessary, split it into incremental commits/patches (within the same task) and keep each step verifiable.

## Tests
- Always run the test suite locally before finishing a task: `python -m pytest`.
- Add or update tests when changing behavior.

## Docs
- Update `README.md` whenever behavior, CLI, formats, or defaults change.

