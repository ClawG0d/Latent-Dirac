"""FastAPI AI gateway: NL + engine schema -> candidate scene JSON.

The client (desktop app) posts the user's prompt, the engine's Scene JSON
Schema (from GET /schema), optionally the current scene and a prior
validation error. The gateway calls Claude with the schema as a forced
tool so the model emits a scene object, and returns it for the client to
validate locally. Engine-version-agnostic: the schema always comes from
the caller, never hardcoded here.

The Anthropic client is injectable (`create_app(client=...)`) so tests
run with no network and no anthropic package; in production the default
client reads ANTHROPIC_API_KEY from the environment.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "claude-sonnet-5"
_TOOL_NAME = "emit_scene"

SYSTEM_PROMPT = (
    "You design Latent Dirac simulation scenes for an antimatter-factory "
    "simulator. Given a natural-language request, emit ONE scene object via "
    "the emit_scene tool, matching the provided JSON schema exactly. Rules:\n"
    "- Use only source/element/solver types allowed by the schema; invalid "
    "keys or types are rejected by fail-fast validation.\n"
    "- Honesty discipline: every model has a fidelity tier; do not claim more "
    "than the parameterized/surrogate/table-based models provide, and prefer "
    "always-available elements. Elements that need an external engine "
    "(matter_slab needs a Geant4 build; xsuite_lattice needs xtrack) should "
    "only be used when the request clearly calls for them.\n"
    "- Give every element and the source a unique label; labels become the "
    "loss-ledger stage names.\n"
    "- If a prior validation error is provided, fix exactly what it reports."
)


class GenerateRequest(BaseModel):
    # `scene_schema` carries the engine's Scene JSON Schema; the JSON key stays
    # "schema" via the alias (a bare `schema` field would shadow BaseModel)
    model_config = ConfigDict(populate_by_name=True)

    prompt: str
    scene_schema: dict = Field(alias="schema")
    current_scene: dict | None = None
    validation_error: list | dict | None = None


def _default_client():  # pragma: no cover - requires the anthropic package + key
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _build_messages(req: GenerateRequest) -> list[dict]:
    parts = [f"Request: {req.prompt}"]
    if req.current_scene is not None:
        parts.append(f"Current scene (edit this): {req.current_scene}")
    if req.validation_error is not None:
        parts.append(
            "The previous attempt failed validation with these errors; correct "
            f"them: {req.validation_error}"
        )
    return [{"role": "user", "content": "\n\n".join(parts)}]


def create_app(client=None) -> FastAPI:
    app = FastAPI(title="Latent Dirac AI gateway")

    def _client():
        return client if client is not None else _default_client()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        tool = {
            "name": _TOOL_NAME,
            "description": "Emit one Latent Dirac scene matching the schema.",
            "input_schema": req.scene_schema,
        }
        response = _client().messages.create(
            model=os.environ.get("GATEWAY_MODEL", DEFAULT_MODEL),
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=_build_messages(req),
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
                return {"scene": block.input}
        return JSONResponse(
            status_code=502,
            content={"detail": "the model returned no scene; retry or refine the prompt"},
        )

    return app
