# LLM Collaboration - BFCL v4

This repo provides BFCL function-calling environments for CoMLRL. The current
layout follows the task-separated style used by the Minecraft collaboration
repos: each BFCL task directory is a self-contained path with its own config,
dataset loader, formatter, parser, logger, reward, and training entrypoints.
The two task directories do not import each other and there is no root shared
BFCL runtime layer.

## Dataset

The task datasets are split on Hugging Face:

- `OpenMLRL/BFCL-V4-Parallel-Native`
- `OpenMLRL/BFCL-V4-Parallel-Multi-Turn`

Both are derived from official BFCL v4 data.

Default configs follow the writing repo's split-slice style and use the first
half of each selected train/eval split via `dataset.train_split` and
`dataset.eval_split`, for example `train[:160]` and `eval[:40]` for native
parallel. Set those fields back to `train` and `eval` to use the full splits.

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
- MAGRPO config: `native_parallel/configs/native_parallel_magrpo_config.yaml`
- MAGRPO entrypoint: `native_parallel/train/train_magrpo.py`
- additional native entrypoints:
  - `native_parallel/train/train_iac.py`
  - `native_parallel/train/train_maac.py`
  - `native_parallel/train/train_madpo.py`
  - `native_parallel/train/train_marlhf.py`
  - `native_parallel/train/train_madpo_iter.py`
  - `native_parallel/train/train_marlhf_iter.py`
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

Both tasks use two Qwen3-4B-Instruct-2507 agents, `self_select` decentralized prompting, no
LoRA or quantization. Native MAGRPO, MAAC, IAC, MADPO, and MARLHF defaults are
aligned to roughly 2560 logged environment steps on the first half of the
non-live native train split. Native MAGRPO, MADPO, MARLHF, MADPO-Iter, and
MARLHF-Iter use `joint_mode: aligned`, because the CoMLRL preference pair
generation path only supports aligned candidates.
MAAC defaults to a third GPU for its shared critic, and MARLHF defaults to a
third GPU for the learned reward model.

Native MAAC/IAC use one generation per prompt and 16 training epochs. With
`rollout_buffer_size: 4` and `train_batch_size: 4`, they keep the same 2560
environment-step budget and the same total number of actor-critic updates while
avoiding the larger per-prompt generation footprint from four return sequences.

Preference defaults follow the Code Generation CHE settings where possible
while preserving BFCL's non-iter step budget. Non-iter MADPO and MARLHF use 80
candidates and select 16 reward-gap pairs per sample. MADPO counts two joint
responses per DPO pair under the BFCL step accounting, while MARLHF counts
online rollout joint responses separately from preference pairs. Iterative
MADPO/MARLHF use 20 current candidates, 20 comparator candidates, 4 selected
pairs per sample, `pair_selection: comparator_reward`, and lambda replay with
`preference_replay_lambda: 0.8`. With 4 iterations, MADPO-Iter counts two joint
responses per DPO pair, while MARLHF-Iter uses two online epochs with four
aligned generations; both are 1280 environment steps per iteration and 5120
total.
There is intentionally no root training entrypoint or root BFCL helper package;
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
do not train; they load the same Qwen3-4B-Instruct-2507 model, let one agent answer the eval
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

## Centralized MAGRPO

`native_parallel/train/train_magrpo.py` supports
`--override magrpo.collaboration_mode=centralized magrpo.num_turns=1`.
It trains one joint-input/joint-output actor using the task reward, without a
comparator, preference dataset, or learned reward model. The BFCL adapter splits
role outputs for rewards and evaluation. `max_new_tokens` is the total
joint-response budget. The default remains decentralized. Set
`agent_model.attn_implementation=sdpa` when testing the memory-efficient attention
backend.

## Centralized Preference Collaboration

The native-parallel MADPO, MARLHF, and iterative trainers support one trainable
model generating all task roles. The default remains decentralized. Enable it
with:

```bash
python native_parallel/train/train_madpo_iter.py --config native_parallel/configs/native_parallel_madpo_iter_config.yaml --override madpo_iter.collaboration_mode=centralized
python native_parallel/train/train_marlhf_iter.py --config native_parallel/configs/native_parallel_marlhf_iter_config.yaml --override marlhf_iter.collaboration_mode=centralized
```

Use the `madpo` / `marlhf` prefix for the non-iterative scripts. Keep
`num_agents=2` for task roles; `agent_model` is loaded once. Explicit `agents`
and actor device lists must describe one model. `BFCLCentralizedComparatorAdapter`
combines both original function-calling prompts and extracts `<agent_0>` /
`<agent_1>` outputs only for task rewards and evaluation. Policy training,
preference replay, and learned reward scoring retain the full joint text.
Iterative comparators automatically use centralized generation (actor index 0)
for current, history, external model, and API sources.

`max_new_tokens` limits the whole joint response; consider doubling the previous
per-role budget. MARLHF `reward_max_length` must fit the joint prompt and all
responses. Reward/comparator devices remain independently configurable. Existing
decentralized behavior and step counters are unchanged.

## Slurm runtime cache

Training launchers isolate unset CUDA JIT caches per job on node-local storage.
Update CoMLRL alongside this checkout; see the [runtime cache guide](docs/runtime_cache.md).
