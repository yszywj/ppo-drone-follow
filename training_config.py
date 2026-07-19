from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


SPECIAL_OBJECTS = {
    ("task", "motion_pool"),
}


def load_training_config(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    raw = _load_raw_config(config_path, ())

    flattened: Dict[str, Any] = {}
    special: Dict[str, Any] = {}
    for section, section_value in raw.items():
        if section in ("version", "extends"):
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
    return flattened, special, raw


def _load_raw_config(
    config_path: Path,
    ancestors: Tuple[Path, ...],
) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"training config does not exist: {config_path}")
    if config_path in ancestors:
        cycle = " -> ".join(str(path) for path in (*ancestors, config_path))
        raise ValueError(f"training config extends cycle: {cycle}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("training config root must be a JSON object")
    extends = raw.get("extends")
    if extends is None:
        return copy.deepcopy(dict(raw))
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("training config extends must be a non-empty path string")
    base_path = Path(extends).expanduser()
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    base = _load_raw_config(base_path, (*ancestors, config_path))
    override = {key: value for key, value in raw.items() if key != "extends"}
    merged = _deep_merge(base, override)
    merged["extends"] = str(base_path)
    return merged


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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
