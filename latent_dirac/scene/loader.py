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


def _resolve_scene_relative_paths(data, base_dir: Path) -> None:
    """Resolve file-referencing source params against the scene file's directory.

    A relative `table_path` in a scene file means "relative to the scene
    file", so scenes keep working regardless of the process cwd.
    Mapping-based scenes (`scene_from_mapping`) have no file anchor and
    keep plain-path semantics.
    """

    source = data.get("source") if isinstance(data, Mapping) else None
    params = source.get("params") if isinstance(source, Mapping) else None
    if isinstance(params, dict):
        table_path = params.get("table_path")
        if isinstance(table_path, str) and not Path(table_path).is_absolute():
            params["table_path"] = str((base_dir / table_path).resolve())

    elements = data.get("elements") if isinstance(data, Mapping) else None
    for element in elements if isinstance(elements, list) else []:
        if not isinstance(element, dict):
            continue
        if element.get("type") == "xsuite_lattice":
            line_path = element.get("line_path")
            if isinstance(line_path, str) and not Path(line_path).is_absolute():
                element["line_path"] = str((base_dir / line_path).resolve())
        elif element.get("type") == "buffer_gas_cooling":
            cross_section_path = element.get("cross_section_path")
            if isinstance(cross_section_path, str) and not Path(cross_section_path).is_absolute():
                element["cross_section_path"] = str((base_dir / cross_section_path).resolve())


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
    _resolve_scene_relative_paths(data, file_path.resolve().parent)
    return scene_from_mapping(data)
