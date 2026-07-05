"""Load scenes from YAML or JSON documents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from latent_dirac.scene.schema import Scene

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


def scene_from_mapping(data: Mapping) -> Scene:
    return Scene.model_validate(data)


def load_scene(path: str | Path) -> Scene:
    """Load a scene file; the format is detected from the file suffix."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")

    if suffix in _YAML_SUFFIXES:
        data = yaml.safe_load(text)
    elif suffix in _JSON_SUFFIXES:
        data = json.loads(text)
    else:
        raise ValueError(
            f"unsupported scene file suffix {suffix!r}; expected one of "
            f"{sorted(_YAML_SUFFIXES | _JSON_SUFFIXES)}"
        )
    return scene_from_mapping(data)
