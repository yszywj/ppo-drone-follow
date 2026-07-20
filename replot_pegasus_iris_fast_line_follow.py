#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
RUNPY_ROOT = SCRIPT_DIR.parent
if str(RUNPY_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNPY_ROOT))

from pegasus_iris_fast_line_follow.ppo_core import save_training_plots


def parse_csv_value(value: str) -> Any:
    stripped = value.strip()
    if stripped == "True":
        return True
    if stripped == "False":
        return False
    if not stripped:
        return ""
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return value


def load_metrics_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            {key: parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        "Regenerate fast Pegasus PPO plots from saved CSV metrics"
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    update_rows = load_metrics_rows(run_dir / "metrics" / "update_metrics.csv")
    episode_rows = load_metrics_rows(run_dir / "metrics" / "episode_metrics.csv")
    if not update_rows and not episode_rows:
        raise FileNotFoundError(f"no metric rows found below {run_dir / 'metrics'}")
    save_training_plots(run_dir, update_rows, episode_rows)
    print(
        f"[FAST PPO] regenerated plots from {len(update_rows)} updates and "
        f"{len(episode_rows)} episodes: {run_dir / 'plots'}"
    )


if __name__ == "__main__":
    main()
