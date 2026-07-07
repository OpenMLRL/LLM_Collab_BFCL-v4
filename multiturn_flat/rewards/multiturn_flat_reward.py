"""Order-sensitive reward for flattened BFCL multi-turn steps."""

from __future__ import annotations

from typing import Any, Dict

from rewards.bfcl_rewards import BFCLReward, BFCLRewardConfig


def make_reward(config: Dict[str, Any] | None = None) -> BFCLReward:
    """Build the flattened multi-turn reward.

    This task scores the aggregate as an ordered sequence: agent 1 calls followed
    by agent 2 calls, compared against the current turn's ordered BFCL gold calls.
    """
    raw = dict(config or {})
    raw["mode"] = "sequence_flat"
    return BFCLReward(BFCLRewardConfig.from_dict(raw))
