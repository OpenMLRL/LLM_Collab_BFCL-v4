"""Train MAGRPO on native BFCL parallel and live-parallel tasks."""

from __future__ import annotations

import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_magrpo import main


def _add_default_config() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(
            [
                "--config",
                str(TASK_ROOT / "configs" / "native_parallel_magrpo_config.yaml"),
            ]
        )


if __name__ == "__main__":
    _add_default_config()
    main()
