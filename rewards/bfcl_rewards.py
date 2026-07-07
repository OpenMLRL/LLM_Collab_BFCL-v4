"""Joint BFCL reward with collaboration penalties and balance shaping."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from bfcl.parsing import (
    ToolCall,
    canonical_call_key,
    canonical_value,
    dedupe_calls,
    gold_to_tool_calls,
    parse_tool_calls,
)


def _value_matches(predicted: Any, accepted: Any) -> bool:
    if isinstance(accepted, list):
        return any(_value_matches(predicted, item) for item in accepted)
    if isinstance(predicted, list) and isinstance(accepted, tuple):
        return canonical_value(predicted) == canonical_value(list(accepted))
    pred_norm = canonical_value(predicted)
    accepted_norm = canonical_value(accepted)
    if isinstance(pred_norm, (int, float)) and isinstance(accepted_norm, (int, float)):
        return abs(float(pred_norm) - float(accepted_norm)) < 1e-6
    return pred_norm == accepted_norm


def _call_similarity(predicted: ToolCall, gold: ToolCall) -> Tuple[float, bool]:
    if predicted.name != gold.name:
        return 0.0, False
    if not gold.arguments:
        exact = not predicted.arguments
        return (1.0 if exact else 0.7), exact

    matched = 0
    for arg_name, accepted_values in gold.arguments.items():
        if arg_name not in predicted.arguments:
            continue
        if _value_matches(predicted.arguments[arg_name], accepted_values):
            matched += 1

    base = matched / max(1, len(gold.arguments))
    extra_args = max(0, len(set(predicted.arguments) - set(gold.arguments)))
    if extra_args:
        base *= len(gold.arguments) / (len(gold.arguments) + extra_args)
    exact = matched == len(gold.arguments) and extra_args == 0
    return float(base), bool(exact)


def _greedy_match(
    predicted_calls: Sequence[ToolCall], gold_calls: Sequence[ToolCall]
) -> Tuple[float, int, int]:
    candidates = []
    for pred_idx, predicted in enumerate(predicted_calls):
        for gold_idx, gold in enumerate(gold_calls):
            score, exact = _call_similarity(predicted, gold)
            if score > 0:
                candidates.append((score, exact, pred_idx, gold_idx))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    used_pred = set()
    used_gold = set()
    score_sum = 0.0
    exact_count = 0
    matched_count = 0
    for score, exact, pred_idx, gold_idx in candidates:
        if pred_idx in used_pred or gold_idx in used_gold:
            continue
        used_pred.add(pred_idx)
        used_gold.add(gold_idx)
        score_sum += float(score)
        exact_count += int(exact)
        matched_count += 1

    denom = max(1, len(gold_calls))
    return score_sum / denom, exact_count, matched_count


def _function_f1(predicted_calls: Sequence[ToolCall], gold_calls: Sequence[ToolCall]) -> float:
    pred_counts = Counter(call.name for call in predicted_calls)
    gold_counts = Counter(call.name for call in gold_calls)
    overlap = sum(min(pred_counts[name], gold_counts[name]) for name in gold_counts)
    pred_total = sum(pred_counts.values())
    gold_total = sum(gold_counts.values())
    if pred_total == 0 or gold_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / gold_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ideal_counts(num_calls: int, num_agents: int) -> List[int]:
    base = num_calls // num_agents
    extra = num_calls % num_agents
    return [base + (1 if idx < extra else 0) for idx in range(num_agents)]


def _balance_score(agent_call_counts: Sequence[int], gold_count: int) -> float:
    if not agent_call_counts or gold_count <= 0:
        return 0.0
    ideal = _ideal_counts(gold_count, len(agent_call_counts))
    diff = sum(abs(int(actual) - expected) for actual, expected in zip(agent_call_counts, ideal))
    return max(0.0, 1.0 - diff / max(1, gold_count))


def _cross_agent_overlap(agent_calls: Sequence[Sequence[ToolCall]]) -> int:
    key_counts = Counter()
    for calls in agent_calls:
        for key in {canonical_call_key(call) for call in calls}:
            key_counts[key] += 1
    return sum(max(0, count - 1) for count in key_counts.values())


@dataclass
class BFCLRewardConfig:
    mode: str = "auto"
    parse_weight: float = 0.05
    function_weight: float = 0.25
    argument_weight: float = 0.35
    count_weight: float = 0.10
    balance_weight: float = 0.15
    exact_bonus: float = 0.20
    overlap_penalty: float = 0.20
    lazy_agent_penalty: float = 0.15
    extra_call_penalty: float = 0.05
    sequence_function_weight: float = 0.15
    sequence_argument_weight: float = 0.20
    sequence_order_weight: float = 0.25
    sequence_prefix_weight: float = 0.10
    sequence_count_weight: float = 0.10
    sequence_balance_weight: float = 0.15
    sequence_exact_bonus: float = 0.20
    sequence_overlap_penalty: float = 0.15
    sequence_lazy_agent_penalty: float = 0.25
    sequence_extra_call_penalty: float = 0.15
    sequence_duplicate_penalty: float = 0.05
    min_reward: float = -0.4
    max_reward: float = 1.2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BFCLRewardConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        values = {key: value for key, value in (raw or {}).items() if key in allowed}
        return cls(**values)


@dataclass
class BFCLReward:
    config: BFCLRewardConfig = field(default_factory=BFCLRewardConfig)
    last_details: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, *agent_completions, batch_items=None, prompts=None) -> List[float]:
        del prompts
        if batch_items is None:
            raise ValueError("BFCLReward requires batch_items for ground-truth calls.")
        rewards = []
        details = []
        for sample_idx, batch_item in enumerate(batch_items):
            completions = []
            for agent_output in agent_completions:
                if isinstance(agent_output, str):
                    completions.append(agent_output if sample_idx == 0 else "")
                elif sample_idx < len(agent_output):
                    completions.append(agent_output[sample_idx])
                else:
                    completions.append("")
            reward, detail = score_bfcl_joint_response(
                completions,
                batch_item=batch_item,
                config=self.config,
            )
            rewards.append(reward)
            details.append(detail)
        self.last_details = details
        return rewards


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _uses_sequence_reward(batch_item: Dict[str, Any], cfg: BFCLRewardConfig) -> bool:
    mode = str(cfg.mode or "auto").lower()
    if mode in {"sequence", "sequence_flat", "multi_turn", "multi_turn_flat"}:
        return True
    if mode in {"flat", "single_turn"}:
        return False
    category = str(batch_item.get("official_category", ""))
    support_level = str(batch_item.get("support_level", ""))
    source_view = str(batch_item.get("source_view", ""))
    return (
        support_level == "experimental_flattened_sequence_step"
        or source_view == "multi_turn_step_flattened"
        or category.startswith("multi_turn_")
    )


def _sequence_alignment_score(
    predicted_calls: Sequence[ToolCall], gold_calls: Sequence[ToolCall]
) -> float:
    if not gold_calls:
        return 0.0
    if not predicted_calls:
        return 0.0

    rows = len(predicted_calls) + 1
    cols = len(gold_calls) + 1
    dp = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i, predicted in enumerate(predicted_calls, start=1):
        for j, gold in enumerate(gold_calls, start=1):
            similarity, _ = _call_similarity(predicted, gold)
            dp[i][j] = max(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1] + similarity,
            )
    return min(1.0, dp[-1][-1] / max(1, len(gold_calls)))


def _prefix_score(
    predicted_calls: Sequence[ToolCall], gold_calls: Sequence[ToolCall]
) -> float:
    if not gold_calls or not predicted_calls:
        return 0.0
    score = 0.0
    for predicted, gold in zip(predicted_calls, gold_calls):
        similarity, _ = _call_similarity(predicted, gold)
        if similarity <= 0.0:
            break
        score += similarity
    return min(1.0, score / max(1, len(gold_calls)))


def _ordered_exact_match(
    predicted_calls: Sequence[ToolCall], gold_calls: Sequence[ToolCall]
) -> bool:
    if len(predicted_calls) != len(gold_calls) or not gold_calls:
        return False
    for predicted, gold in zip(predicted_calls, gold_calls):
        _, exact = _call_similarity(predicted, gold)
        if not exact:
            return False
    return True


def _duplicate_rate(calls: Sequence[ToolCall], gold_count: int) -> float:
    if not calls:
        return 0.0
    counts = Counter(canonical_call_key(call) for call in calls)
    duplicate_count = sum(max(0, count - 1) for count in counts.values())
    return min(1.0, duplicate_count / max(1, gold_count))


def score_bfcl_joint_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
    config: BFCLRewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    cfg = config or BFCLRewardConfig()
    if _uses_sequence_reward(batch_item, cfg):
        return score_bfcl_sequence_flat_response(
            agent_completions,
            batch_item=batch_item,
            config=cfg,
        )

    return score_bfcl_flat_joint_response(
        agent_completions,
        batch_item=batch_item,
        config=cfg,
    )


def score_bfcl_flat_joint_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
    config: BFCLRewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    cfg = config or BFCLRewardConfig()
    function_schemas = batch_item.get("function", []) or []
    gold_calls = gold_to_tool_calls(batch_item.get("ground_truth", []) or [])

    per_agent_calls = [
        dedupe_calls(parse_tool_calls(text or "", function_schemas=function_schemas))
        for text in agent_completions
    ]
    combined_calls = dedupe_calls(
        call for calls in per_agent_calls for call in calls
    )

    parse_score = 1.0 if combined_calls else 0.0
    function_f1 = _function_f1(combined_calls, gold_calls)
    argument_score, exact_count, matched_count = _greedy_match(combined_calls, gold_calls)
    gold_count = len(gold_calls)
    pred_count = len(combined_calls)
    count_score = max(0.0, 1.0 - abs(pred_count - gold_count) / max(1, gold_count))

    agent_counts = [len(calls) for calls in per_agent_calls]
    balance_score = _balance_score(agent_counts, gold_count)
    overlap_count = _cross_agent_overlap(per_agent_calls)
    overlap_rate = min(1.0, overlap_count / max(1, gold_count))
    lazy_agents = sum(1 for count in agent_counts if count == 0)
    lazy_rate = lazy_agents / max(1, len(agent_counts))
    extra_calls = max(0, pred_count - gold_count)
    extra_rate = min(1.0, extra_calls / max(1, gold_count))

    exact_match = (
        exact_count == gold_count
        and pred_count == gold_count
        and gold_count > 0
    )

    reward = (
        cfg.parse_weight * parse_score
        + cfg.function_weight * function_f1
        + cfg.argument_weight * argument_score
        + cfg.count_weight * count_score
        + cfg.balance_weight * balance_score
        + (cfg.exact_bonus if exact_match else 0.0)
        - cfg.overlap_penalty * overlap_rate
        - cfg.lazy_agent_penalty * lazy_rate
        - cfg.extra_call_penalty * extra_rate
    )
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    detail = {
        "reward": float(reward),
        "exact_match": float(exact_match),
        "parse_success": parse_score,
        "function_f1": float(function_f1),
        "argument_match": float(argument_score),
        "matched_calls": float(matched_count),
        "exact_calls": float(exact_count),
        "pred_call_count": float(pred_count),
        "gold_call_count": float(gold_count),
        "count_score": float(count_score),
        "balance_score": float(balance_score),
        "overlap_count": float(overlap_count),
        "overlap_rate": float(overlap_rate),
        "lazy_agents": float(lazy_agents),
        "lazy_rate": float(lazy_rate),
        "extra_call_rate": float(extra_rate),
        "agent_call_counts": agent_counts,
        "combined_calls": [
            {"name": call.name, "arguments": call.arguments} for call in combined_calls
        ],
        "reward_mode": 0.0,
    }
    return float(reward), detail


def score_bfcl_sequence_flat_response(
    agent_completions: Sequence[str],
    *,
    batch_item: Dict[str, Any],
    config: BFCLRewardConfig | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Score flattened BFCL multi-turn steps with order-sensitive shaping.

    The aggregate order is deterministic: agent 1 calls followed by agent 2 calls.
    This gives decentralized agents a learnable signal to split sequential work
    into earlier and later chunks while still using the existing flat trainer.
    """
    cfg = config or BFCLRewardConfig()
    function_schemas = batch_item.get("function", []) or []
    gold_calls = gold_to_tool_calls(batch_item.get("ground_truth", []) or [])

    per_agent_calls = [
        parse_tool_calls(text or "", function_schemas=function_schemas)
        for text in agent_completions
    ]
    combined_ordered = [
        call for calls in per_agent_calls for call in calls
    ]

    parse_score = 1.0 if combined_ordered else 0.0
    function_f1 = _function_f1(combined_ordered, gold_calls)
    argument_score, exact_count, matched_count = _greedy_match(
        combined_ordered, gold_calls
    )
    sequence_score = _sequence_alignment_score(combined_ordered, gold_calls)
    prefix_score = _prefix_score(combined_ordered, gold_calls)
    gold_count = len(gold_calls)
    pred_count = len(combined_ordered)
    count_score = max(0.0, 1.0 - abs(pred_count - gold_count) / max(1, gold_count))

    agent_counts = [len(calls) for calls in per_agent_calls]
    balance_score = _balance_score(agent_counts, gold_count)
    overlap_count = _cross_agent_overlap(per_agent_calls)
    overlap_rate = min(1.0, overlap_count / max(1, gold_count))
    lazy_agents = sum(1 for count in agent_counts if count == 0)
    lazy_rate = lazy_agents / max(1, len(agent_counts))
    extra_calls = max(0, pred_count - gold_count)
    extra_rate = min(1.0, extra_calls / max(1, gold_count))
    duplicate_rate = _duplicate_rate(combined_ordered, gold_count)
    exact_match = _ordered_exact_match(combined_ordered, gold_calls)

    reward = (
        cfg.parse_weight * parse_score
        + cfg.sequence_function_weight * function_f1
        + cfg.sequence_argument_weight * argument_score
        + cfg.sequence_order_weight * sequence_score
        + cfg.sequence_prefix_weight * prefix_score
        + cfg.sequence_count_weight * count_score
        + cfg.sequence_balance_weight * balance_score
        + (cfg.sequence_exact_bonus if exact_match else 0.0)
        - cfg.sequence_overlap_penalty * overlap_rate
        - cfg.sequence_lazy_agent_penalty * lazy_rate
        - cfg.sequence_extra_call_penalty * extra_rate
        - cfg.sequence_duplicate_penalty * duplicate_rate
    )
    reward = _clamp(reward, cfg.min_reward, cfg.max_reward)

    detail = {
        "reward": float(reward),
        "exact_match": float(exact_match),
        "parse_success": parse_score,
        "function_f1": float(function_f1),
        "argument_match": float(argument_score),
        "matched_calls": float(matched_count),
        "exact_calls": float(exact_count),
        "pred_call_count": float(pred_count),
        "gold_call_count": float(gold_count),
        "count_score": float(count_score),
        "balance_score": float(balance_score),
        "overlap_count": float(overlap_count),
        "overlap_rate": float(overlap_rate),
        "lazy_agents": float(lazy_agents),
        "lazy_rate": float(lazy_rate),
        "extra_call_rate": float(extra_rate),
        "sequence_score": float(sequence_score),
        "prefix_score": float(prefix_score),
        "duplicate_rate": float(duplicate_rate),
        "agent_call_counts": agent_counts,
        "combined_calls": [
            {"name": call.name, "arguments": call.arguments}
            for call in combined_ordered
        ],
        "reward_mode": 1.0,
    }
    return float(reward), detail


def make_bfcl_reward(config: Dict[str, Any] | None = None) -> BFCLReward:
    return BFCLReward(BFCLRewardConfig.from_dict(config or {}))
