# LLM Collaboration - BFCL v4

This repo provides BFCL function-calling environments for CoMLRL. The initial
implementation focuses on decentralized two-agent MAGRPO for BFCL v4
single-turn `parallel` and `parallel_multiple` tasks.

## Dataset

The default dataset is
`OpenMLRL/BFCL-v4-Parallel-Categorized`. It is derived from the official BFCL v4
non-live single-turn `parallel` and `parallel_multiple` files in
`ShishirPatil/gorilla`.

Added fields:

- `official_category`: `parallel` or `parallel_multiple`
- `task_type`: OpenMLRL heuristic keyword category for filtering
- `internal_split`: deterministic 160/40 train/eval split within each official
  category, stratified by `task_type`

`task_type` is not an official BFCL label.

## MAGRPO

Default setup:

- two Qwen3-8B agents
- agent 0 on `cuda:0`, agent 1 on `cuda:1`
- no LoRA or quantization
- single-turn MAGRPO with `joint_mode: cross`
- BFCL v4 non-live and live `parallel` / `parallel_multiple` categories
- self-selected decentralized roles by default
- 2 training epochs

The default formatter is `bfcl.role_mode: self_select`: each agent is told
another agent is helping, that it does not need to solve the whole request, and
that it should contribute a useful non-empty subset.

Run:

```bash
python3 train_magrpo.py --config configs/magrpo_bfcl_v4_config.yaml
```

Filter to one BFCL category:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override dataset.categories='["parallel"]'
```

Include experimental flattened multi-turn candidates:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override dataset.categories='["parallel","parallel_multiple","live_parallel","live_parallel_multiple","multi_turn_base_step"]'
```

Run only the experimental flattened multi-turn candidates:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override dataset.categories='["multi_turn_base_step","multi_turn_long_context_step","multi_turn_miss_func_step","multi_turn_miss_param_step"]'
```

Filter by heuristic task type:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override dataset.task_types='["travel/local_services/logistics"]'
```

Use fixed first-half/second-half roles:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override bfcl.role_mode=split_by_order
```

## Reward

The joint reward parses each agent's tool calls, aggregates them, and compares
the merged action against BFCL ground truth. It includes:

- parse success
- function-name F1
- argument-value match
- call-count match
- exact-match bonus
- overlap penalty for duplicated calls across agents
- lazy-agent penalty when an agent emits no calls
- balance reward for keeping each agent's contribution close to an even split

`bfcl_reward.mode: auto` uses the original flat aggregate reward for native
single-turn parallel rows. For flattened multi-turn rows, it switches to an
order-sensitive sequence reward: agent outputs are aggregated in agent-index
order, then scored against the current turn's ordered gold calls with sequence,
prefix, count, balance, overlap, lazy-agent, extra-call, and duplicate-call
terms. This is a better fit for the experimental multi-turn candidates, but it
is still not a full BFCL stateful environment with actual tool execution.
