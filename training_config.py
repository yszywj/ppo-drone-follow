from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


SPECIAL_OBJECTS = {
    ("task", "motion_pool"),
}


def load_training_config(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("training config root must be a JSON object")

    flattened: Dict[str, Any] = {}
    special: Dict[str, Any] = {}
    for section, section_value in raw.items():
        if section == "version":
            continue
        if not isinstance(section_value, Mapping):
            _insert_unique(flattened, str(section), section_value, str(section))
            continue
        for key, value in section_value.items():
            location = (str(section), str(key))
            if location in SPECIAL_OBJECTS:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{section}.{key} must be a JSON object")
                special[key] = dict(value)
                continue
            if isinstance(value, Mapping):
                raise ValueError(
                    f"unsupported nested config object at {section}.{key}; "
                    "only task.motion_pool may be nested"
                )
            _insert_unique(flattened, str(key), value, f"{section}.{key}")

    flattened["config"] = str(config_path)
    return flattened, special, dict(raw)


def resolve_path_from_config(config_path: str | Path, value: str) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(config_path).expanduser().resolve().parent / path).resolve())


def _insert_unique(target: Dict[str, Any], key: str, value: Any, location: str) -> None:
    if key in target:
        raise ValueError(f"duplicate config key {key!r}; repeated at {location}")
    target[key] = value
