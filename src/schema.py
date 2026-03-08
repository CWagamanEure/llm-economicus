from dataclasses import dataclass
from typing import Any, Literal, Mapping, NotRequired, TypedDict


class ExpectedValueAssumptions(TypedDict):
    probabilities_are_known: bool
    utility_model: Literal["linear"]
    decision_rule: Literal["expected_value_maximization"]
    tie_epsilon: float


class TimeDiscountingAssumptions(TypedDict):
    discount_model: Literal["simple"]
    annual_discount_rate: float
    tie_epsilon: float


class CertainOption(TypedDict):
    type: Literal["certain"]
    amount: int | float


class LotteryOption(TypedDict):
    type: Literal["lottery"]
    p_win: float
    win_amount: int
    lose_amount: int


class CertaintyEquivalentLotteryOption(TypedDict):
    type: Literal["lottery"]
    p_win: float
    high: int
    low: int


class MixedLotteryOption(TypedDict):
    type: Literal["mixed_lottery"]
    p_gain: float
    gain: int
    loss: int


class ImmediateOption(TypedDict):
    type: Literal["immediate"]
    amount: int
    delay_days: Literal[0]


class DelayedOption(TypedDict):
    type: Literal["delayed"]
    amount: float
    delay_days: int


class LotteryOptions(TypedDict):
    A: CertainOption
    B: LotteryOption


class CertaintyEquivalentOptions(TypedDict):
    A: CertainOption
    B: CertaintyEquivalentLotteryOption


class MixedGainLossOptions(TypedDict):
    A: CertainOption
    B: MixedLotteryOption


class TimeDiscountingOptions(TypedDict):
    A: ImmediateOption
    B: DelayedOption


class LotteryProblemSpec(TypedDict):
    task_subtype: Literal["lottery_choice"]
    objective: str
    options: LotteryOptions
    assumptions: ExpectedValueAssumptions


class CeOfferComparisonProblemSpec(TypedDict):
    task_subtype: Literal["ce_offer_comparison"]
    objective: str
    options: CertaintyEquivalentOptions
    assumptions: ExpectedValueAssumptions


class MixedGainLossProblemSpec(TypedDict):
    task_subtype: Literal["mixed_gain_loss_choice"]
    objective: str
    options: MixedGainLossOptions
    assumptions: ExpectedValueAssumptions


class TimeDiscountingProblemSpec(TypedDict):
    task_subtype: Literal["time_discounting"]
    objective: str
    options: TimeDiscountingOptions
    assumptions: TimeDiscountingAssumptions


class BayesianAssumptions(TypedDict):
    prior_high: float
    p_signal_high_given_high: float
    p_signal_high_given_low: float
    observed_signal: Literal["high", "low"]
    signal_model: Literal["binary_conditional_likelihood"]
    decision_rule: Literal["bayes_update_then_expected_value"]
    tie_epsilon: float
    posterior_threshold: NotRequired[float]
    public_actions: NotRequired[list[Literal["choose_high", "choose_low"]]]
    transaction_cost: NotRequired[float]


class StateHypothesisOption(TypedDict):
    type: Literal["state_hypothesis"]
    state: Literal["high", "low"]


class BinaryActOption(TypedDict):
    type: Literal["act"]
    payoff_if_high: int | float
    payoff_if_low: int | float


class BinaryDoNotActOption(TypedDict):
    type: Literal["do_not_act"]
    payoff: int | float


class CascadeActionOption(TypedDict):
    type: Literal["cascade_action"]
    implied_state: Literal["high", "low"]


class AssetBuyOption(TypedDict):
    type: Literal["buy_asset"]
    value_if_high: int | float
    value_if_low: int | float
    market_price: int | float


class AssetNoBuyOption(TypedDict):
    type: Literal["do_not_buy"]
    payoff: int | float


class BasicBayesUpdateOptions(TypedDict):
    A: StateHypothesisOption
    B: StateHypothesisOption


class BinarySignalDecisionOptions(TypedDict):
    A: BinaryActOption
    B: BinaryDoNotActOption


class InformationCascadeOptions(TypedDict):
    A: CascadeActionOption
    B: CascadeActionOption


class NoisySignalAssetUpdateOptions(TypedDict):
    A: AssetBuyOption
    B: AssetNoBuyOption


class BasicBayesUpdateProblemSpec(TypedDict):
    task_subtype: Literal["basic_bayes_update"]
    objective: str
    options: BasicBayesUpdateOptions
    assumptions: BayesianAssumptions


class BinarySignalDecisionProblemSpec(TypedDict):
    task_subtype: Literal["binary_signal_decision"]
    objective: str
    options: BinarySignalDecisionOptions
    assumptions: BayesianAssumptions


class InformationCascadeProblemSpec(TypedDict):
    task_subtype: Literal["information_cascade_step"]
    objective: str
    options: InformationCascadeOptions
    assumptions: BayesianAssumptions


class NoisySignalAssetUpdateProblemSpec(TypedDict):
    task_subtype: Literal["noisy_signal_asset_update"]
    objective: str
    options: NoisySignalAssetUpdateOptions
    assumptions: BayesianAssumptions


ProblemSpec = (
    LotteryProblemSpec
    | CeOfferComparisonProblemSpec
    | MixedGainLossProblemSpec
    | TimeDiscountingProblemSpec
    | BasicBayesUpdateProblemSpec
    | BinarySignalDecisionProblemSpec
    | InformationCascadeProblemSpec
    | NoisySignalAssetUpdateProblemSpec
)

RiskLossTimeTaskSubtype = Literal[
    "lottery_choice",
    "ce_offer_comparison",
    "mixed_gain_loss_choice",
    "time_discounting",
]

BayesianSignalTaskSubtype = Literal[
    "basic_bayes_update",
    "binary_signal_decision",
    "information_cascade_step",
    "noisy_signal_asset_update",
]

TaskSubtype = RiskLossTimeTaskSubtype | BayesianSignalTaskSubtype


class ComparisonPair(TypedDict):
    left_action: str
    right_action: str


class SolverTrace(TypedDict):
    left_action: str
    right_action: str
    left_value: float
    right_value: float
    tie_epsilon: float
    comparison_result: str


class ActionScalars(dict[str, float]):
    """Validated per-action scalar map used for action evaluation."""

    def __init__(self, values: Mapping[str, float] | None = None):
        super().__init__()
        if values is None:
            return
        for action, value in values.items():
            self[action] = value

    def __setitem__(self, action: str, value: float) -> None:
        if not isinstance(action, str):
            raise TypeError("Action key must be a string.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Action scalar for '{action}' must be numeric.")
        super().__setitem__(action, float(value))


@dataclass
class Target:
    objective: str
    state: dict[str, Any]
    beliefs: dict[str, Any]
    constraints: dict[str, Any]
    actions: list[str]
    # Ordered action pair used for direct two-option comparisons.
    comparison_pair: ComparisonPair
    outcome_model: dict[str, Any]
    # Comparable per-action scalar values used by the decision rule.
    # For risk/loss tasks these are expected utilities; for time-discounting
    # tasks these are discounted present values.
    action_values: ActionScalars
    # Numeric values directly compared to pick the optimal decision.
    decision_values: ActionScalars
    optimal_decision: str
    # Minimal structured record of the comparison that produced the decision.
    solver_trace: SolverTrace
    brief_rationale: str

    def __post_init__(self) -> None:
        left_action = self.comparison_pair.get("left_action")
        right_action = self.comparison_pair.get("right_action")
        if not isinstance(left_action, str) or not isinstance(right_action, str):
            raise ValueError(
                "comparison_pair must include string left_action and right_action."
            )
        self.action_values = ActionScalars(self.action_values)
        self.decision_values = ActionScalars(self.decision_values)


@dataclass
class Metadata:
    generator_name: str
    # Generator/dataset schema version used to produce this sample.
    version: str
    # Seed used to initialize the generator RNG state; not a per-example seed.
    seed: int
    # Role of this datapoint within the broader dataset pipeline.
    dataset_role: str
    # Prompt style requested when initializing the generator (e.g., "random").
    requested_prompt_style: str | None
    # Prompt style actually used to render this sample.
    resolved_prompt_style: str | None
    # Whether rendered prompt text includes explicit action labels/tokens.
    prompt_has_action_labels: bool
    # Deterministic fingerprint for this sample's core structured content.
    example_fingerprint: str | None
    # Tie threshold used by the decision rule for this sample.
    tie_threshold: float | None
    # Per-sample diagnostics for difficulty calibration.
    difficulty_metrics: dict[str, Any]
    # Monotonic index of the generated sample within a generator instance.
    sample_index: int | None = None


@dataclass
class DataPoint:
    task_family: str
    task_subtype: TaskSubtype
    task_id: str
    difficulty: str
    # Canonical structured problem representation used to render the prompt.
    problem_spec: ProblemSpec
    input: str
    target: Target
    metadata: Metadata
