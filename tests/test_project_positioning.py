from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml"}
FORBIDDEN_TERMS = ("Ge" + "nesis-style", "Ge" + "nesis World", "gene" + "sis_world")
ALLOWED_ADAPTERS = {"geant4", "root", "xsuite"}


def iter_project_text_files():
    for path in PROJECT_ROOT.rglob("*"):
        if path == Path(__file__).resolve():
            continue
        if any(part in {".git", ".pytest_cache", ".venv", "__pycache__"} for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_corrected_positioning_removes_old_platform_wording():
    offenders = []
    for path in iter_project_text_files():
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} contains {term}")

    assert offenders == []


def test_only_corrected_placeholder_adapters_are_present():
    adapter_dirs = {
        path.name
        for path in (PROJECT_ROOT / "latent_dirac" / "adapters").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert adapter_dirs == ALLOWED_ADAPTERS
