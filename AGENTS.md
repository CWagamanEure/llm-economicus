# AGENTS.md

This file defines operating instructions for any coding agent working in this repository.

## 1) Mission

Build and maintain a dataset generator for economic rationality tasks, currently focused on risk/loss/time-choice scenarios.

Primary goals:
- Keep generated data schema-consistent and deterministic under fixed seeds.
- Preserve normative decision logic (expected value, discounting, prospect-value + constraints).
- Keep tests fast and meaningful.

## 2) Repository Map

- `src/schema.py`
  - Core dataclasses: `DataPoint`, `Target`, `Metadata`.
- `src/data-generation/base_generator.py`
  - Abstract base generator and metadata helper.
- `src/data-generation/risk_loss_time_generator.py`
  - Main generator implementation for task subtypes:
    - `lottery_choice`
    - `certainty_equivalent`
    - `mixed_gain_loss_choice`
    - `prospect_gain_loss`
    - `time_discounting`
- `tests/test_risk_loss_time_generator.py`
  - Unit tests for task dispatch, metadata behavior, tie behavior, formulas, and bankruptcy logic.
- `README.md`
  - Research framing, hypotheses, and intended benchmark/evaluation direction.

Ignore `__pycache__` files and never rely on them.

## 3) Environment and Commands

This project uses Poetry and Python `>=3.11,<4.0`.

Install dependencies:

```bash
poetry install
```

Run tests:

```bash
poetry run pytest
```

Run lint checks:

```bash
poetry run ruff check .
```

Optional formatting check (if formatting rules are added later):

```bash
poetry run ruff format --check .
```

## 4) Code and Design Rules

### 4.1 Schema-first changes

When changing generated data content:
- Keep `DataPoint` / `Target` / `Metadata` contracts coherent.
- Ensure every generated sample has:
  - clear `objective`
  - explicit `actions` including `"indifferent"` when ties are possible
  - reproducible `expected_utilities`
  - concise, verifiable `brief_rationale`

### 4.2 Determinism and randomness

- Use the generator's local RNG (`self.rng`), not global `random.*`.
- Preserve seed semantics:
  - `metadata.seed` = generator initialization seed (`base_seed`)
  - `metadata.sample_index` increments monotonically per generated sample

### 4.3 Decision logic

- Tie handling must remain explicit via `"indifferent"`.
- Keep objective and outcome model aligned with utility calculations.
- For prospect tasks, maintain:
  - reference-dependent value function parameters
  - explicit bankruptcy/no-negative-final-wealth constraint
  - penalty override behavior used in expected utilities

### 4.4 Difficulty labels

- Difficulty is currently a static mapping in `DEFAULT_DIFFICULTY_BY_SUBTYPE`.
- If upgrading to dynamic difficulty, update tests and document logic.

## 5) Testing Expectations

For any behavior change, add or adjust tests in `tests/`.

Minimum checks before finishing:
1. `poetry run pytest` passes.
2. `poetry run ruff check .` passes.
3. New/changed logic has targeted unit coverage.

When fixing a bug:
- Add a regression test that fails before the fix and passes after.

## 6) Common Pitfalls

- The source folder `src/data-generation` contains a hyphen; imports in tests currently use `sys.path` insertion. Do not "clean this up" opportunistically unless explicitly asked, because it can break test/import behavior.
- Avoid broad refactors in the same change as logic fixes.
- Do not silently change task text templates if tests or downstream consumers depend on formula strings.

## 7) Change Workflow for Agents

1. Read relevant files and existing tests first.
2. Make the smallest correct change.
3. Update/add tests.
4. Run lint + tests.
5. Summarize:
   - what changed
   - why
   - how validated
   - any assumptions or residual risk

## 8) Definition of Done

A task is done when:
- requested behavior is implemented,
- tests and lint pass locally,
- schema and metadata invariants are preserved,
- and the change summary is clear enough for another engineer to review quickly.

