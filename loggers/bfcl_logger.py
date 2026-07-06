"""Evaluation logging adapters for CoMLRL's MAGRPO trainer."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from rewards.bfcl_rewards import BFCLRewardConfig, score_bfcl_joint_response


def _prompt_key(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


def build_bfcl_eval_logger(
    eval_rows: Iterable[Dict[str, Any]],
    *,
    reward_config: Dict[str, Any] | None = None,
) -> Callable[..., List[Dict[str, Any]]]:
    row_by_prompt = {
        _prompt_key(row.get("prompt") or row.get("user_prompt") or ""): row
        for row in eval_rows
    }
    cfg = BFCLRewardConfig.from_dict(reward_config or {})

    def logger(
        agent_completions_turns: List[List[List[str]]],
        test_cases: List[str],
        entry_points: List[str],
        prompts: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        del test_cases, entry_points
        if not agent_completions_turns or prompts is None:
            return []
        num_samples = len(prompts)
        metrics = []
        for sample_idx in range(num_samples):
            row = row_by_prompt.get(_prompt_key(prompts[sample_idx]))
            if row is None:
                continue
            num_turns = max(
                len(agent_completions_turns[agent_idx][sample_idx])
                for agent_idx in range(len(agent_completions_turns))
            )
            sample_metrics: Dict[str, Any] = {
                "sample_id": row.get("id", sample_idx),
                "official_category": row.get("official_category", ""),
                "task_type": row.get("task_type", ""),
            }
            for turn_idx in range(num_turns):
                turn_completions = []
                for agent_idx in range(len(agent_completions_turns)):
                    per_sample = agent_completions_turns[agent_idx][sample_idx]
                    turn_completions.append(
                        per_sample[turn_idx] if turn_idx < len(per_sample) else ""
                    )
                _, detail = score_bfcl_joint_response(
                    turn_completions,
                    batch_item=row,
                    config=cfg,
                )
                prefix = f"turn_{turn_idx + 1}"
                for key, value in detail.items():
                    if isinstance(value, (int, float)):
                        sample_metrics[f"{prefix}/{key}"] = float(value)
            metrics.append(sample_metrics)
        return metrics

    return logger


def aggregate_bfcl_metrics_for_logging(
    metrics_list: List[Dict[str, Any]], num_turns: int = 1
) -> Dict[str, float]:
    if not metrics_list:
        return {}

    aggregated: Dict[str, float] = {}
    metric_names = [
        "reward",
        "exact_match",
        "parse_success",
        "function_f1",
        "argument_match",
        "matched_calls",
        "exact_calls",
        "pred_call_count",
        "gold_call_count",
        "count_score",
        "balance_score",
        "overlap_count",
        "overlap_rate",
        "lazy_agents",
        "lazy_rate",
        "extra_call_rate",
    ]
    for turn in range(1, num_turns + 1):
        for metric_name in metric_names:
            key = f"turn_{turn}/{metric_name}"
            values = [
                float(sample[key])
                for sample in metrics_list
                if key in sample and isinstance(sample[key], (int, float))
            ]
            if values:
                aggregated[f"turn_{turn}/{metric_name}"] = float(np.mean(values))

    # Official categories and heuristic task types are useful eval slices.
    for group_key in ("official_category", "task_type"):
        group_values = sorted(
            {str(sample.get(group_key, "")) for sample in metrics_list if sample.get(group_key)}
        )
        for group_value in group_values:
            subset = [
                sample
                for sample in metrics_list
                if str(sample.get(group_key, "")) == group_value
            ]
            exact_values = [
                sample["turn_1/exact_match"]
                for sample in subset
                if "turn_1/exact_match" in sample
            ]
            if exact_values:
                safe_name = group_value.replace("/", "_")
                aggregated[f"{group_key}/{safe_name}/exact_match"] = float(
                    np.mean(exact_values)
                )
    return aggregated
