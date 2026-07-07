"""Flat aggregate reward for native BFCL parallel function-calling tasks."""

from __future__ import annotations

from typing import Any, Dict

from rewards.bfcl_rewards import BFCLReward, BFCLRewardConfig


def make_reward(config: Dict[str, Any] | None = None) -> BFCLReward:
    """Build the native single-turn parallel reward.

    This task always uses the flat aggregate scorer: agent calls are deduped,
    merged, and compared as an unordered set against BFCL gold calls.
    """
    raw = dict(config or {})
    raw["mode"] = "flat"
    return BFCLReward(BFCLRewardConfig.from_dict(raw))
