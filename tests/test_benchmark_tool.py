"""Benchmark tool smoke: structure and labels, tiny sizes, CPU-safe.

The committed docs/benchmarks.md comes only from a full run on the
GPU box; this test just pins the generator's contract.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("jax")


def test_quick_run_produces_labeled_document(tmp_path, monkeypatch):
    from tools import run_benchmarks

    output = tmp_path / "bench.md"
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--quick", "--output", str(output)])
    assert run_benchmarks.main() == 0

    text = output.read_text(encoding="utf-8")
    assert "# Benchmarks" in text
    assert "## Environment" in text
    assert "jax" in text and "numpy" in text
    # the honesty labels the discipline requires
    for label in ("dt =", "B =", "parameterized tier", "Boris"):
        assert label in text, label
    assert "particle·steps/s" in text
