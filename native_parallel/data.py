"""Dataset loading for native BFCL parallel and live-parallel data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset
from huggingface_hub import hf_hub_download


DEFAULT_NATIVE_CATEGORIES = (
    "parallel",
    "parallel_multiple",
    "live_parallel",
    "live_parallel_multiple",
)


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
    split_path = path / f"{split}.jsonl" if path.is_dir() else path
    if not split_path.exists():
        raise FileNotFoundError(f"Native BFCL split file not found: {split_path}")
    return _read_jsonl(split_path)


def _load_rows(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    if Path(dataset_name).exists():
        return _load_local_dataset(dataset_name, split)
    if "/" in dataset_name and split in {"train", "eval", "validation", "test"}:
        try:
            downloaded = hf_hub_download(
                repo_id=dataset_name,
                filename=f"{split}.jsonl",
                repo_type="dataset",
            )
            return _read_jsonl(Path(downloaded))
        except Exception:
            pass
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _filter_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    categories: Sequence[str],
    task_types: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    category_set = set(categories)
    invalid = sorted(category_set - set(DEFAULT_NATIVE_CATEGORIES))
    if invalid:
        raise ValueError(
            "Native parallel categories must be one of "
            f"{list(DEFAULT_NATIVE_CATEGORIES)}; got {invalid}."
        )
    task_type_set = set(task_types) if task_types else None

    filtered = []
    for row in rows:
        if row.get("official_category") not in category_set:
            continue
        if task_type_set and row.get("task_type") not in task_type_set:
            continue
        item = dict(row)
        item["prompt"] = item.get("user_prompt", "")
        item.setdefault("test", "")
        item.setdefault("entry_point", "")
        filtered.append(item)
        if max_samples is not None and len(filtered) >= int(max_samples):
            break
    return filtered


def load_native_parallel_dataset(
    dataset_name: str,
    *,
    split: str,
    categories: Optional[Any] = None,
    task_types: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = _load_rows(dataset_name, split)
    selected_categories = _as_list(categories) or list(DEFAULT_NATIVE_CATEGORIES)
    filtered = _filter_rows(
        rows,
        categories=selected_categories,
        task_types=_as_list(task_types),
        max_samples=max_samples,
    )
    if not filtered:
        raise ValueError(
            "Native BFCL filter produced no rows. Check dataset.categories, "
            "dataset.task_types, and split settings."
        )
    return filtered
