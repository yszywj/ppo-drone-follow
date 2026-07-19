#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _latest_metrics(case_dir: Path, filename: str) -> Path | None:
    candidates = list(case_dir.glob(f"seed*/metrics/{filename}"))
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _read(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(case_dir: Path) -> dict[str, object]:
    episodes = _read(_latest_metrics(case_dir, "episode_metrics.csv"))
    updates = _read(_latest_metrics(case_dir, "update_metrics.csv"))
    reasons = Counter(row.get("done_reason", "unknown") for row in episodes)

    def episode_mean(key: str) -> float:
        return fmean(_float(row, key) for row in episodes) if episodes else 0.0

    def update_mean(key: str) -> float:
        return fmean(_float(row, key) for row in updates) if updates else 0.0

    return {
        "case": case_dir.name,
        "episodes": len(episodes),
        "success_rate": (
            sum(row.get("success", "").lower() == "true" for row in episodes)
            / len(episodes)
            if episodes
            else 0.0
        ),
        "moving_joint": episode_mean("moving_good_fraction"),
        "moving_xy": episode_mean("moving_xy_good_fraction"),
        "moving_velocity": episode_mean("moving_velocity_good_fraction"),
        "moving_z": episode_mean("moving_z_good_fraction"),
        "mean_xy_error": update_mean("mean_xy_err"),
        "saturation": update_mean("cmd_saturation_fraction"),
        "local_quality": update_mean(
            "mean_local_tracking_soft_joint_quality"
        ),
        "reasons": json.dumps(dict(reasons), sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        nargs="?",
        default=(
            "/home/1234/workspace/runpy/pegasus_iris_fast_line_follow/"
            "result/ablation"
        ),
    )
    args = parser.parse_args()
    root = Path(args.results_root).expanduser()
    rows = [
        summarize(case_dir)
        for case_dir in sorted(root.iterdir())
        if case_dir.is_dir()
    ]
    fields = (
        "case",
        "episodes",
        "success_rate",
        "moving_joint",
        "moving_xy",
        "moving_velocity",
        "moving_z",
        "mean_xy_error",
        "saturation",
        "local_quality",
        "reasons",
    )
    print("\t".join(fields))
    for row in rows:
        print(
            "\t".join(
                f"{row[field]:.4f}"
                if isinstance(row[field], float)
                else str(row[field])
                for field in fields
            )
        )


if __name__ == "__main__":
    main()
