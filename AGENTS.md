# AGENTS.md

This file defines operating instructions for any coding agent working in this repository.

## 1) Mission

Build and maintain a dataset generator for economic rationality and behavioral-decision tasks across multiple families:
- `risk_loss_time_choice`
- `bayesian_signal_extraction`
- `belief_bias_judgment`

Primary goals:
- Keep generated data schema-consistent and deterministic under fixed seeds.
- Preserve normative decision logic (EV/discounting, Bayesian updating, probability/independence logic, interval coverage logic).
- Keep tests fast and meaningful.

## 2) Repository Map

- `src/schema.py`
  - Core dataclasses: `DataPoint`, `Target`, `Metadata`.
- `src/data-generation/base_generator.py`
  - Abstract base generator and metadata helper.
- `src/data-generation/risk_loss_time_generator.py`
  - Generator for risky choice and intertemporal tasks.
- `src/data-generation/bayesian_signal_generator.py`
  - Generator for Bayesian signal extraction and posterior decision tasks.
- `src/data-generation/belief_bias_generator.py`
  - Generator for belief-bias judgment tasks.
- `tests/test_risk_loss_time_generator.py`
  - Unit tests for risk/loss/time generation and solver behavior.
- `tests/test_bayesian_signal_generator.py`
  - Unit tests for Bayesian generation/rendering/validation behavior.
- `tests/test_belief_bias_generator.py`
  - Unit tests for belief-bias generation/rendering/validation behavior.
- `tests/test_prompt_rendering_contracts.py`
  - Cross-family prompt rendering invariants and leakage checks.

Risk/loss/time subtypes include:
    - `lottery_choice`
    - `certainty_equivalent`
    - `mixed_gain_loss_choice`
    - `prospect_gain_loss`
    - `time_discounting`
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
  - reproducible `action_values`/`decision_values` under the task's normative model
  - concise, verifiable `brief_rationale`

### 4.2 Determinism and randomness

- Use the generator's local RNG (`self.rng`), not global `random.*`.
- Preserve seed semantics:
  - `metadata.seed` = generator initialization seed (`base_seed`)
  - `metadata.sample_index` increments monotonically per generated sample

### 4.3 Decision logic

- Tie handling must remain explicit via `"indifferent"`.
- Keep objective and outcome model aligned with utility calculations.
- For risk/prospect tasks, maintain:
  - reference-dependent value function parameters
  - explicit bankruptcy/no-negative-final-wealth constraint
  - penalty override behavior used in expected utilities
- For Bayesian tasks, maintain prior/likelihood/update consistency and posterior-payoff action scoring.
- For belief-bias tasks, preserve logical semantics (conjunction subset relation, independence claims, binomial-tail comparisons, interval coverage objective).

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
- Do not silently change task text templates if tests or downstream consumers depend on wording contracts, validation linting, or metadata fields (`prompt_style_regime`, `prompt_frame_variant`, `semantic_context`, etc.).

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
