# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Latent Dirac local sim engine (the desktop sidecar).
#
# Build (run on each target OS — a frozen binary is platform-specific):
#   cd desktop/packaging
#   python -m PyInstaller engine.spec --noconfirm
# Output: dist/latent-dirac-engine/  (one-folder build; the executable is
# dist/latent-dirac-engine/latent-dirac-engine[.exe]). electron-builder then
# bundles that folder as an extraResource (see desktop/package.json).
#
# One-folder (not one-file) on purpose: faster startup and no per-launch temp
# extraction, and it drops straight into electron-builder's extraResources.
#
# Lean by construction: only the engine's real import closure ships — numpy,
# pydantic, pyyaml, fastapi, uvicorn, plotly. The heavy optional stacks are
# excluded, and the vendored Geant4 tree is never a Python import so it can
# never be pulled in.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = [
    # uvicorn resolves these dynamically at runtime
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    # our own server package (imported lazily inside __main__.main)
    "latent_dirac.server",
    "latent_dirac.server.app",
]

# plotly ships its JS bundle as package data; include_plotlyjs=True reads it to
# inline the runtime into the offline 3D HTML, so it must travel with the binary
datas = collect_data_files("plotly")
hiddenimports += collect_submodules("plotly")

excludes = [
    # the optional heavy stacks the engine does not need for source->acceptance
    "jax",
    "jaxlib",
    "openpmd_api",
    "uproot",
    "awkward",
    "xsuite",
    "xtrack",
    "xpart",
    "xdeps",
    "xobjects",
    # never needed at runtime
    "matplotlib",
    "tkinter",
    "IPython",
    "pytest",
]

a = Analysis(
    ["engine_entry.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="latent-dirac-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # required: the "PORT <n>" line is read from stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="latent-dirac-engine",
)
