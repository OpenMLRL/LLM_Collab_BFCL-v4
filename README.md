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
- agent 0 handles the first half of ordered tool-call intents
- agent 1 handles the second half

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

Filter by heuristic task type:

```bash
python3 train_magrpo.py \
  --config configs/magrpo_bfcl_v4_config.yaml \
  --override dataset.task_types='["travel/local_services/logistics"]'
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
- balance reward for splitting call counts close to the first-half/second-half assignment
