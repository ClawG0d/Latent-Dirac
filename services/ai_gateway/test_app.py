"""Tests for the AI gateway (NL -> scene JSON) with a mocked LLM client.

The gateway is a standalone deployable that turns a natural-language
prompt + the engine's Scene JSON Schema into a candidate scene via a
Claude tool call. The Anthropic client is injectable, so these tests run
with no network and no anthropic package installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from services.ai_gateway.app import create_app  # noqa: E402


class _FakeToolUse:
    type = "tool_use"
    name = "emit_scene"

    def __init__(self, scene):
        self.input = scene


class _FakeMessages:
    def __init__(self, scene, recorder):
        self._scene = scene
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)

        class _Resp:
            content = [_FakeToolUse(self._scene)]

        return _Resp()


class _FakeClient:
    """Stands in for anthropic.Anthropic; records the last create() kwargs."""

    def __init__(self, scene):
        self.last_call: dict = {}
        self.messages = _FakeMessages(scene, self.last_call)


CANNED_SCENE = {
    "schema_version": 1,
    "name": "generated",
    "seed": 1,
    "solver": {"dt_s": 4e-12, "steps": 40},
    "source": {"type": "positron_pair", "label": "src", "params": {}},
    "elements": [{"type": "monitor", "label": "end"}],
}

SCHEMA = {"title": "Scene", "type": "object", "properties": {"elements": {}}}


def make_client(scene=CANNED_SCENE):
    fake = _FakeClient(scene)
    return TestClient(create_app(client=fake)), fake


def test_generate_returns_scene_from_tool_use():
    client, _ = make_client()
    resp = client.post("/generate", json={"prompt": "a positron beam", "schema": SCHEMA})
    assert resp.status_code == 200
    assert resp.json()["scene"] == CANNED_SCENE


def test_tool_input_schema_is_the_request_schema():
    client, fake = make_client()
    client.post("/generate", json={"prompt": "x", "schema": SCHEMA})
    tools = fake.last_call["tools"]
    assert len(tools) == 1
    assert tools[0]["input_schema"] == SCHEMA
    # the model is forced to emit the scene via the tool
    assert fake.last_call["tool_choice"]["name"] == tools[0]["name"]


def test_prompt_and_system_vocabulary_are_sent():
    client, fake = make_client()
    client.post("/generate", json={"prompt": "trap some antiprotons", "schema": SCHEMA})
    system = fake.last_call["system"]
    # the system prompt teaches the vocabulary + honesty discipline
    assert "fidelity" in system.lower()
    messages = fake.last_call["messages"]
    joined = " ".join(str(m) for m in messages)
    assert "trap some antiprotons" in joined


def test_validation_error_is_fed_back_for_retry():
    client, fake = make_client()
    errors = [{"loc": ["elements", 0, "type"], "msg": "unknown element 'warp'"}]
    client.post(
        "/generate",
        json={"prompt": "x", "schema": SCHEMA, "validation_error": errors},
    )
    messages = fake.last_call["messages"]
    joined = " ".join(str(m) for m in messages)
    assert "warp" in joined  # the prior error is included so the model corrects


def test_current_scene_context_is_included():
    client, fake = make_client()
    client.post(
        "/generate",
        json={"prompt": "add an aperture", "schema": SCHEMA, "current_scene": CANNED_SCENE},
    )
    joined = " ".join(str(m) for m in fake.last_call["messages"])
    assert "generated" in joined  # the current scene name appears in context


def test_model_is_configurable(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODEL", "claude-test-model")
    client, fake = make_client()
    client.post("/generate", json={"prompt": "x", "schema": SCHEMA})
    assert fake.last_call["model"] == "claude-test-model"


def test_missing_tool_use_returns_502():
    # if the model returns no tool_use block, surface a clean error
    class _EmptyMessages:
        def create(self, **kwargs):
            class _Resp:
                content = []

            return _Resp()

    class _EmptyClient:
        messages = _EmptyMessages()

    client = TestClient(create_app(client=_EmptyClient()))
    resp = client.post("/generate", json={"prompt": "x", "schema": SCHEMA})
    assert resp.status_code == 502
