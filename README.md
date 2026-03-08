# LLM Economicus


## Question

- Does fine-tuning an LLM on rational economic behavior produce more rational behavior, as measured by stable improvements in utility-consistent decisions, reduced behavioral biases, and better performance in downstream economic environments?


## Hypothesis

1. Fine-tuning an LLM on normative economic decision data will move models to benchmark rational behavior in tasks like risk, loss, time and social preference elicitation.

2. The "rationality gain" will be more substantive when the training data is structured as explicit decision schema rather than plain natural-language economic explanations, because structured outputs make the learned process more algorithmic and less narrative. 


## Testing

1. Measure whether the model's preferences and beliefs became more utility consistent.
2. Test with prompt and context/task changes.
3. Test in real environments.

## Terms

1. **Rationality** - We need a definition for normative economic behavior. We can define economic rationality as properly applying utility theory and bayesian updating to changes in state.

- **Benchmark A**: Expected utility maximization 
- **Benchmark B**: Bayesian rationality (not updating salient or recent info) 
- **Benchmark C**: Economic strategic rationality
- **Benchmark D**: Financial Rationality (properly identifying and acting on positive EV) 

## Generate Mock Data

Run from repo root:

```bash
poetry run python main.py --count 100 --seed 42 --output data/mock_risk_loss_time.jsonl
```

Useful options:

- `--prompt-style` one of `default`, `formal`, `plain_english`, `compact`, `finance_framed`, `unlabeled`, `random`
- `--version` metadata version tag (default `v1`)

## Training Data

- General Structure:

```json
{
  "objective": "maximize expected utility",
  "state": {...},
  "beliefs": {...},
  "actions": [...],
  "comparison_pair": {"left_action": "...", "right_action": "..."},
  "outcome_model": {...},
  "action_values": {...},
  "constraints": {...},
  "optimal_decision": "...",
  "rationale": "brief formal explanation"
}
```

`action_values` stores comparable action values under the task's normative model.
In risk/loss tasks these are expected utilities; in time-discounting tasks these are discounted present values.

- Reproducibility metadata:
  - Each generated sample includes `metadata.seed` and `metadata.version`.
  - `metadata.seed` is the generator initialization seed (shared across samples in one run), not a per-example seed.
  - `metadata.sample_index` identifies the sample order within that seeded run.
  - `problem_spec.assumptions.tie_epsilon` records the tie-threshold used for decision comparisons.
  - Prompt rendering supports multiple styles (`default`, `formal`, `plain_english`, `compact`, `finance_framed`) via `RiskLossTimeGenerator(prompt_style=...)`; `prompt_style="random"` samples a style deterministically from the generator RNG.

- For **market tasks**, we extend with:

```json
{
  "valuation": ...,
  "price_target": ...,
  "order_type": "...",
  "quantity": ...
}
```

5 main buckets:

1. **Risk / loss / time choice tasks**

- lotteries
- certainty equivalents
- prospect-style gain/loss problems
- time discounting problems

2. **Strategic game tasks**

- ultimatum
- dictator
- prisoner's dilemma
- public goods
- second-price auctions

3. Bayesian updating and signal extraction

- basic prior/liklihood/posterior tasks
- information cascades
- noisy signal trading and fundamental-value updates

4. Financial decision tasks

- arbitrage under frictions
- portfolio allocation under volatility and frictions
- market making under inventory constraints
- valuation vs current price
- limit vs market order choice

5. Bias-counterexample tasks

- sunk cost
- anchoring
- framing
- endowment effect
- disposition effect / reference dependence


## Comparisons

- Compare outputs from:

1. Base model
2. Prompted base model
3. SFT on plain economics explanations
4. SFT on structured normative data
5. SFT on structured normative data and rationale traces
6. Maybe RL


## Eval

- **Metrics**:

1. deviation from normative optimum
2. certainty-equivalent error
3. inferred utility-parameter shift
4. frequency of dominance violations
5. transitivity violations
6. framing sensitivity
7. within-condition variance

- **Benchmarks (per model)**:

1. paraphrases
2. reordered answer options
3. advice  vs acting for self
4. different temperatures
5. unseen numeric ranges
6. unseen tasks

- **Game benchmarks (single agent and multi)**:

1. treasury alocation
2. trade/no-trade
3. hedge/no-hedge
4. arbitration under transaction costs
5. DAO decisions


## Failure modes

- only improves near training distribution
- more brittle under paraphrasing
- more deterministic
- over optimizes in one benchmark
- less context-sensitive 
