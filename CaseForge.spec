# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).resolve()
ICON_PATH = ROOT / "assets" / "icon.ico"
ASSETS_PATH = ROOT / "assets"


a = Analysis(
    [str(ROOT / "benchmark_case_generator" / "gui.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ASSETS_PATH), "assets")],
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "setuptools"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CaseForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_PATH),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipfiles,
    a.dependencies,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CaseForge",
)
