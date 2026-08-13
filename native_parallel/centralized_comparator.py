"""Centralized comparator adapter for native BFCL tool calling."""

from typing import Any, Dict, Sequence

from comlrl.trainers.preference import (
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)


class BFCLCentralizedComparatorAdapter:
    def __init__(self) -> None:
        self._tagged_parser = TaggedCentralizedComparatorAdapter()

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        del batch_item
        prompt_sections = "\n\n".join(
            f"Agent {agent_idx} original prompt:\n{prompt}"
            for agent_idx, prompt in enumerate(agent_prompts)
        )
        output_sections = "\n".join(
            f"<agent_{agent_idx}>\nfunction calls for Agent {agent_idx}, or []\n"
            f"</agent_{agent_idx}>"
            for agent_idx in range(len(agent_prompts))
        )
        return f"""You are acting as one centralized coordinator for {len(agent_prompts)} function-calling agents.

Generate the exact separate output each decentralized agent would submit. Respect
each agent's assignment from its original prompt. Across all outputs, cover the
requested tool-call intents without unnecessary duplication. Inside each section,
use only the Python-style function calls required by that agent, one call per line,
or [] when that agent has no valid call. Do not add explanations or markdown.

{prompt_sections}

Return exactly this structure:
{output_sections}
"""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        try:
            return self._tagged_parser.parse_completion(
                completion,
                batch_item,
                num_agents,
            )
        except CentralizedComparatorParseError:
            return [""] * num_agents

    def build_sequential_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
        agent_index: int,
        previous_outputs: Sequence[str],
    ) -> str:
        del batch_item
        if agent_index < 0 or agent_index >= len(agent_prompts):
            raise ValueError("agent_index must identify one BFCL agent prompt.")
        if len(previous_outputs) != agent_index:
            raise ValueError(
                "Sequential BFCL generation must follow increasing agent index."
            )

        prompt_sections = "\n\n".join(
            f"Agent {idx} original assignment:\n{prompt}"
            for idx, prompt in enumerate(agent_prompts)
        )
        if previous_outputs:
            previous_sections = "\n\n".join(
                f"Final Agent {idx} calls:\n<agent_{idx}>\n{output}\n</agent_{idx}>"
                for idx, output in enumerate(previous_outputs)
            )
        else:
            previous_sections = (
                "No earlier calls have been finalized. Later agents will see your "
                "output."
            )

        return f"""You are Agent {agent_index} in a centralized sequential function-calling team.

The team is producing one joint tool-call action. You can inspect every assignment
and all earlier finalized calls. Produce only Agent {agent_index}'s assigned calls,
covering the remaining required intent without duplicating earlier valid calls.

{prompt_sections}

Earlier finalized calls:
{previous_sections}

Inside the required section, use only Python-style function calls, one call per line,
or [] when this agent has no valid remaining call. Do not add explanations or markdown.
Return exactly this structure:
<agent_{agent_index}>
function calls for Agent {agent_index}, or []
</agent_{agent_index}>
"""

    def parse_sequential_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        agent_index: int,
    ) -> str:
        try:
            return self._tagged_parser.parse_sequential_completion(
                completion,
                batch_item,
                agent_index,
            )
        except CentralizedComparatorParseError:
            return str(completion).strip()
