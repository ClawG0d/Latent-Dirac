"""Xopt-compatible scene evaluator built on the batched JAX backend.

The evaluator follows Xopt's calling convention — a plain callable from an
input dict to an output dict — so any `xopt.Evaluator(function=...)` can
wrap it directly. Xopt itself is not imported or required. The `batch`
method evaluates a whole generation of candidates in one JAX launch, the
intended path for generation-based optimizers.
"""

from __future__ import annotations

import re

import numpy as np

from latent_dirac.backends.jax_scene import BatchedSceneProgram, base_parameters
from latent_dirac.scene.schema import Scene

_VARIABLE_PATTERN = re.compile(r"^(?P<key>[^\[\]]+?)(?:\[(?P<component>\d+)\])?$")


class SceneEvaluator:
    """Evaluate scene objectives as a function of named scalar variables.

    Variables are scalar scene parameters (`"label.param"`) or single
    vector components (`"label.vec[i]"`). Objectives are the loss-ledger
    aggregates: `accepted_fraction` and `accepted_weighted`.
    """

    def __init__(self, scene: Scene, variables: list[str]):
        if not variables:
            raise ValueError("at least one variable is required")

        base = base_parameters(scene)
        self._variables: dict[str, tuple[str, int | None]] = {}
        override_keys: list[str] = []
        for variable in variables:
            match = _VARIABLE_PATTERN.match(variable)
            if match is None:
                raise ValueError(f"variable {variable!r} is not of the form label.param[/index]")
            key = match["key"]
            component = match["component"]
            if key not in base:
                label, _, param = key.partition(".")
                raise ValueError(
                    f"variable {variable!r}: no sweepable parameter {param!r} on any element "
                    f"labeled {label!r} (available: {sorted(base)})"
                )
            base_shape = base[key].shape
            if component is None:
                if base_shape != ():
                    raise ValueError(
                        f"variable {variable!r} refers to a non-scalar parameter of shape "
                        f"{base_shape}; pick a single component like {key}[0]"
                    )
                self._variables[variable] = (key, None)
            else:
                index = int(component)
                if len(base_shape) != 1 or index >= base_shape[0]:
                    raise ValueError(
                        f"variable {variable!r}: component index {index} is out of range "
                        f"for shape {base_shape}"
                    )
                self._variables[variable] = (key, index)
            if key not in override_keys:
                override_keys.append(key)

        self._base = {key: np.asarray(base[key], dtype=float) for key in override_keys}
        self._program = BatchedSceneProgram(scene, override_keys=tuple(override_keys))

    def _compose_values(self, inputs: dict[str, np.ndarray], batch_size: int):
        values = {
            key: np.broadcast_to(base, (batch_size, *base.shape)).copy() for key, base in self._base.items()
        }
        for variable, series in inputs.items():
            key, component = self._variables[variable]
            if component is None:
                values[key][:] = np.reshape(series, (batch_size,))
            else:
                values[key][:, component] = np.reshape(series, (batch_size,))
        return values

    def _validate_keys(self, inputs) -> None:
        missing = set(self._variables) - set(inputs)
        if missing:
            raise ValueError(f"missing variables: {sorted(missing)}")
        unexpected = set(inputs) - set(self._variables)
        if unexpected:
            raise ValueError(f"unexpected variables: {sorted(unexpected)}")

    def __call__(self, inputs: dict[str, float]) -> dict[str, float]:
        self._validate_keys(inputs)
        values = self._compose_values({name: np.asarray([float(value)]) for name, value in inputs.items()}, 1)
        result = self._program.run(values)
        return {
            "accepted_fraction": float(result.accepted_fraction[0]),
            "accepted_weighted": float(result.accepted_weighted[0]),
        }

    def batch(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Evaluate a whole batch of candidates in one JAX launch."""

        self._validate_keys(inputs)
        arrays = {name: np.asarray(series, dtype=float).reshape(-1) for name, series in inputs.items()}
        sizes = {series.shape[0] for series in arrays.values()}
        if len(sizes) != 1:
            raise ValueError(f"all variables must share one batch size, got {sorted(sizes)}")
        batch_size = sizes.pop()
        if batch_size == 0:
            raise ValueError("batch size must be at least 1, got 0")

        result = self._program.run(self._compose_values(arrays, batch_size))
        return {
            "accepted_fraction": result.accepted_fraction,
            "accepted_weighted": result.accepted_weighted,
        }


def make_scene_evaluator(scene: Scene, variables: list[str]) -> SceneEvaluator:
    """Build an Xopt-compatible evaluator for the given scene variables."""

    return SceneEvaluator(scene, variables)
