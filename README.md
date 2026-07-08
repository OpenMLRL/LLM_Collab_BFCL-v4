# LLM Collaboration - BFCL v4

This repo provides BFCL function-calling environments for CoMLRL. The current
layout follows the task-separated style used by the Minecraft collaboration
repos: each BFCL task directory is a self-contained path with its own config,
dataset loader, formatter, parser, logger, reward, and MAGRPO training
entrypoint. The two task directories do not import each other and there is no
root shared BFCL runtime layer.

## Dataset

The task datasets are split on Hugging Face:

- `OpenMLRL/BFCL-V4-Parallel-Native`
- `OpenMLRL/BFCL-V4-Parallel-Multi-Turn`

Both are derived from official BFCL v4 data.

Key fields:

- `official_category`: selectable category.
- `task_type`: OpenMLRL heuristic keyword category for filtering.
- `user_prompt`: prompt text used by the formatter.
- `function`: function schemas shown to agents and used by the parser.
- `ground_truth`: normalized BFCL gold tool calls used by the reward.
- `turn_index`: only present on `multi_turn_*_step` rows; zero-based index in
  the original BFCL multi-turn trajectory.

`task_type` is not an official BFCL label.

## Tasks

### Native Parallel

`native_parallel/` contains the native single-turn BFCL parallel task:

- default categories: `parallel`, `parallel_multiple`
- live categories remain supported via `dataset.categories` overrides:
  `live_parallel`, `live_parallel_multiple`
- reward: flat aggregate joint reward
- dataset: `OpenMLRL/BFCL-V4-Parallel-Native`
- config: `native_parallel/configs/native_parallel_magrpo_config.yaml`
- entrypoint: `native_parallel/train/train_magrpo.py`
- local modules: `data.py`, `formatting.py`, `parsing.py`, `logger.py`,
  `rewards/native_parallel_reward.py`

Run:

```bash
python3 native_parallel/train/train_magrpo.py
```

### Multiturn Flat

`multiturn_flat/` contains flattened BFCL multi-turn step tasks. "Flat" means
each row is one selected current turn from an original BFCL multi-turn
trajectory, converted into a single-turn training item. The row's `user_prompt`
contains the needed history/context, but the gold calls are only for that
current turn. The formatter also surfaces `turn_index` to the agents when this
field is present.

- categories: `multi_turn_base_step`, `multi_turn_long_context_step`,
  `multi_turn_miss_func_step`, `multi_turn_miss_param_step`
- reward: order-sensitive sequence-flat reward
- dataset: `OpenMLRL/BFCL-V4-Parallel-Multi-Turn`
- config: `multiturn_flat/configs/multiturn_flat_magrpo_config.yaml`
- entrypoint: `multiturn_flat/train/train_magrpo.py`
- local modules: `data.py`, `formatting.py`, `parsing.py`, `logger.py`,
  `rewards/multiturn_flat_reward.py`

```bash
python3 multiturn_flat/train/train_magrpo.py
```

Both tasks use two Qwen3-8B agents, `self_select` decentralized prompting, no
LoRA or quantization, `joint_mode: cross`, and 2 training epochs by default.
There is intentionally no root MAGRPO entrypoint or root BFCL helper package;
launch each task through its own `train/` path.

Filter by heuristic task type:

```bash
python3 native_parallel/train/train_magrpo.py \
  --override dataset.task_types='["travel/local_services/logistics"]'
```

Use fixed first-half/second-half roles:

```bash
python3 native_parallel/train/train_magrpo.py \
  --override bfcl.role_mode=split_by_order
```

## Reward

`native_parallel/rewards/native_parallel_reward.py` always uses flat aggregate
scoring: agent calls are deduped, merged, and compared as an unordered set
against BFCL gold calls. It includes:

- parse success
- function-name F1
- argument-value match
- call-count match
- exact-match bonus
- overlap penalty for duplicated calls across agents
- lazy-agent penalty when an agent emits no calls
- balance reward for keeping each agent's contribution close to an even split

`multiturn_flat/rewards/multiturn_flat_reward.py` always uses sequence-flat
scoring: agent outputs are aggregated in agent-index order, then scored against
the current turn's ordered gold calls with sequence, prefix, count, balance,
overlap, lazy-agent, and extra-call terms. Complete duplicate calls from the
same agent are deduped before scoring. This is better for flattened multi-turn
candidates, but it is still not a full BFCL stateful environment with actual
tool execution.

## Baselines

Each task directory has an independent raw single-agent baseline. These scripts
do not train; they load the same Qwen3-8B model, let one agent answer the eval
split, and report the original model's joint `turn_1/exact_match`.

Native parallel:

```bash
python3 native_parallel/baseline/single_agent/eval_single_agent.py
```

Multiturn flat:

```bash
python3 multiturn_flat/baseline/single_agent/eval_single_agent.py
```

Both write `predictions.jsonl` and `summary.json` under their configured output
directory and log the summary to wandb when `wandb.enabled: true`.
