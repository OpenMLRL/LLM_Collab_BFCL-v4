"""Dataset loading for categorized BFCL v4 parallel data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset


def _as_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    raise ValueError(f"Expected a string or list, got {type(value).__name__}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_local_dataset(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    path = Path(dataset_name)
    if path.is_dir():
        split_path = path / f"{split}.jsonl"
    else:
        split_path = path
    if not split_path.exists():
        raise FileNotFoundError(f"BFCL split file not found: {split_path}")
    return _read_jsonl(split_path)


def _load_rows(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    if Path(dataset_name).exists():
        return _load_local_dataset(dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _filter_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    categories: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    category_set = set(categories) if categories else None
    task_type_set = set(task_types) if task_types else None

    filtered = []
    for row in rows:
        if category_set and row.get("official_category") not in category_set:
            continue
        if task_type_set and row.get("task_type") not in task_type_set:
            continue
        row = dict(row)
        row["prompt"] = row.get("user_prompt", "")
        row.setdefault("test", "")
        row.setdefault("entry_point", "")
        filtered.append(row)
        if max_samples is not None and len(filtered) >= int(max_samples):
            break
    return filtered


def load_bfcl_dataset(
    dataset_name: str,
    *,
    split: str,
    categories: Optional[Any] = None,
    task_types: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a BFCL dataset split and apply experiment filters."""
    rows = _load_rows(dataset_name, split)
    filtered = _filter_rows(
        rows,
        categories=_as_list(categories),
        task_types=_as_list(task_types),
        max_samples=max_samples,
    )
    if not filtered:
        raise ValueError(
            "BFCL dataset filter produced no rows. Check dataset.categories, "
            "dataset.task_types, and split settings."
        )
    return filtered
