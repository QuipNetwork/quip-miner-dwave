# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""PyInstaller spec for the self-contained ``quip-dwave-qa`` executable.

The coordinator spawns miners as subprocesses (`[dwave] binary =
"quip-dwave-qa"` in its config), so this ships the same way the CPU, CUDA and
Metal miners do: one file, no Python interpreter, no venv, no Rust toolchain on
the host.

Build from the repository root:

    pyinstaller --clean --noconfirm pyinstaller/quip-dwave-qa.spec

Everything below was derived by building and then running the result, not from
PyInstaller's static analysis alone, which reports success on a binary that
cannot import Ocean.
"""

import os

from PyInstaller.utils.hooks import collect_all

# collect_all() pulls submodules, data files and shared libraries for packages
# whose imports PyInstaller cannot follow statically.
#
# `dwave` is a PEP 420 namespace package with no __init__.py, so each
# subpackage has to be named on its own — collecting "dwave" finds nothing.
# The rest are Ocean's runtime dependencies, imported lazily deep inside
# dwave.cloud and dwave.system; without them the binary builds and then dies
# with ModuleNotFoundError on first use.
_PACKAGES = [
    "dimod",
    "dwave.cloud",
    "dwave.embedding",
    "dwave.preprocessing",
    "dwave.samplers",
    "dwave.system",
    "dwave.optimization",
    "dwave_networkx",
    "minorminer",
    "penaltymodel",
    "homebase",
    "fasteners",
    "diskcache",
    "plucky",
    # The quip_proto SDK is a compiled pyo3 abi3 extension (_core.abi3.so)
    # alongside the generated gRPC stubs.
    "quip_proto",
]

datas, binaries, hiddenimports = [], [], []
for _pkg in _PACKAGES:
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# dimod/cyqmbase/ holds two compiled extensions but has no __init__.py, so
# collect_all's pkgutil walk skips it. Every `import dimod` then fails with
# "No module named 'dimod.cyqmbase.cyqmbase_float32'", which cascades into
# dwave.system, dwave.samplers and dwave.preprocessing. Name them directly.
hiddenimports += [
    "dimod.cyqmbase.cyqmbase_float32",
    "dimod.cyqmbase.cyqmbase_float64",
]

# Plot and dataframe backends that Ocean imports only for optional rendering
# and notebook helpers. The miner never draws anything, and excluding them
# keeps roughly 100 MB of matplotlib/pandas out of the binary.
_EXCLUDES = ["matplotlib", "pandas", "sklearn", "IPython", "tkinter"]

a = Analysis(
    [os.path.join(SPECPATH, "quip_dwave_qa.py")],  # noqa: F821 - PyInstaller global
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    excludes=_EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821 - PyInstaller global

exe = EXE(  # noqa: F821 - PyInstaller global
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="quip-dwave-qa",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    target_arch="arm64",
)
