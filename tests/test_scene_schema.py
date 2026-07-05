import pytest
from pydantic import ValidationError

from latent_dirac.scene.loader import load_scene, scene_from_mapping
from latent_dirac.scene.schema import Scene

VALID_SCENE_YAML = """
schema_version: 1
name: capture-line
seed: 2026
source:
  type: positron_pair
  label: pair-source
  params:
    primary_count: 10000
    yield_eplus_per_primary: 0.02
    mean_energy_MeV: 3.0
    energy_spread_MeV: 0.4
    angular_rms_rad: 0.03
    source_sigma_m: 0.001
    bunch_length_s: 1.0e-12
    macro_particles: 64
solver:
  type: relativistic_boris
  dt_s: 2.0e-12
  steps: 40
elements:
  - { type: solenoid, label: capture-solenoid, b_tesla: 0.8, radius_m: 0.05, length_m: 0.5 }
  - { type: drift, label: gap-1, steps: 10 }
  - { type: aperture, label: collimator, radius_m: 0.04, z_m: 0.06 }
  - { type: momentum_window, label: momentum-cut, p_min_gev_c: 0.001, p_max_gev_c: 0.02 }
  - { type: monitor, label: end-station }
"""


def valid_scene_mapping() -> dict:
    import yaml

    return yaml.safe_load(VALID_SCENE_YAML)


def test_valid_scene_parses_with_all_element_types():
    scene = scene_from_mapping(valid_scene_mapping())

    assert isinstance(scene, Scene)
    assert scene.schema_version == 1
    assert scene.source.type == "positron_pair"
    assert [element.type for element in scene.elements] == [
        "solenoid",
        "drift",
        "aperture",
        "momentum_window",
        "monitor",
    ]
    assert scene.elements[0].label == "capture-solenoid"


def test_yaml_and_json_files_load_identically(tmp_path):
    import json

    yaml_path = tmp_path / "scene.yaml"
    yaml_path.write_text(VALID_SCENE_YAML, encoding="utf-8")
    json_path = tmp_path / "scene.json"
    json_path.write_text(json.dumps(valid_scene_mapping()), encoding="utf-8")

    assert load_scene(yaml_path) == load_scene(json_path)


def test_unknown_element_type_is_rejected():
    mapping = valid_scene_mapping()
    mapping["elements"][0] = {"type": "warp_drive", "label": "nope"}

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_unknown_key_is_rejected_fail_fast():
    mapping = valid_scene_mapping()
    mapping["elements"][0]["fringe_model"] = "tanh"

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_missing_label_is_rejected():
    mapping = valid_scene_mapping()
    del mapping["elements"][0]["label"]

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_duplicate_labels_are_rejected():
    mapping = valid_scene_mapping()
    mapping["elements"][1]["label"] = "capture-solenoid"

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_source_label_collision_with_element_is_rejected():
    mapping = valid_scene_mapping()
    mapping["source"]["label"] = "capture-solenoid"

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_field_element_zero_steps_override_is_rejected_at_load():
    mapping = valid_scene_mapping()
    mapping["elements"][0]["steps"] = 0

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_field_element_negative_steps_override_is_rejected_at_load():
    mapping = valid_scene_mapping()
    mapping["elements"][0]["steps"] = -4

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_wrong_schema_version_is_rejected():
    mapping = valid_scene_mapping()
    mapping["schema_version"] = 2

    with pytest.raises(ValidationError):
        scene_from_mapping(mapping)


def test_unsupported_file_suffix_is_rejected(tmp_path):
    path = tmp_path / "scene.toml"
    path.write_text("x = 1", encoding="utf-8")

    with pytest.raises(ValueError):
        load_scene(path)
