#!/usr/bin/env python3
"""Build the frozen CaseForge directory and a Debian-family .deb package.

This is maintainer tooling. Normal users launch the installed graphical
application from their desktop environment and do not run this script.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "CaseForge.spec"
DEBIAN_SOURCE = ROOT / "debian"
FROZEN_APP = ROOT / "dist" / "CaseForge"
STAGING_ROOT = ROOT / "build" / "caseforge-deb-root"
PACKAGE_OUTPUT = ROOT / "dist"


def _require_repo_venv() -> None:
    expected = (ROOT / ".venv" / "bin" / "python").resolve()
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise SystemExit(
            "CaseForge packaging must run with "
            f"{expected}; received {actual}"
        )


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _remove_generated(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _render_control(version: str) -> str:
    source = (DEBIAN_SOURCE / "control").read_text(encoding="utf-8")
    package_stanza = next(
        (
            stanza
            for stanza in source.split("\n\n")
            if any(line == "Package: caseforge" for line in stanza.splitlines())
        ),
        "",
    )
    if not package_stanza:
        raise ValueError("debian/control does not define the caseforge package")
    if any(line.startswith("Version:") for line in package_stanza.splitlines()):
        raise ValueError("debian/control is a source template; do not duplicate Version")

    lines = package_stanza.splitlines()
    for index, line in enumerate(lines):
        if line == "Package: caseforge":
            lines.insert(index + 1, f"Version: {version}-1")
            break
    else:
        raise ValueError("debian/control does not define the caseforge package")
    return "\n".join(lines).rstrip() + "\n"


def build_package() -> Path:
    """Build and return the real Debian package path."""

    _require_repo_venv()
    from benchmark_case_generator import __version__

    if not SPEC.is_file():
        raise FileNotFoundError(SPEC)

    logo = ROOT / "assets" / "rivet_logo.png"
    if not logo.is_file():
        raise FileNotFoundError(
            "BRANDING ASSET BLOCKED: expected intended asset at " + str(logo)
        )

    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(SPEC),
        ]
    )

    frozen_executable = FROZEN_APP / "CaseForge"
    if not frozen_executable.is_file():
        raise FileNotFoundError(frozen_executable)

    package_root = STAGING_ROOT
    _remove_generated(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    install_root = package_root / "opt" / "caseforge"
    shutil.copytree(FROZEN_APP, install_root)
    installed_executable = install_root / "CaseForge"
    installed_executable.chmod(installed_executable.stat().st_mode | 0o111)

    desktop_target = package_root / "usr" / "share" / "applications" / "caseforge.desktop"
    desktop_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEBIAN_SOURCE / "caseforge.desktop", desktop_target)

    icon_target = (
        package_root
        / "usr"
        / "share"
        / "icons"
        / "hicolor"
        / "256x256"
        / "apps"
        / "caseforge.png"
    )
    icon_target.parent.mkdir(parents=True, exist_ok=True)
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QImage

    source_image = QImage(str(logo))
    if source_image.isNull():
        raise ValueError(f"Unable to read branding asset: {logo}")
    icon_image = source_image.scaled(
        QSize(256, 256),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    if not icon_image.save(str(icon_target), "PNG"):
        raise OSError(f"Unable to write packaged icon: {icon_target}")

    control_target = package_root / "DEBIAN" / "control"
    control_target.parent.mkdir(parents=True, exist_ok=True)
    control_target.write_text(_render_control(__version__), encoding="utf-8")

    package_path = PACKAGE_OUTPUT / f"caseforge_{__version__}_amd64.deb"
    _remove_generated(package_path)
    dpkg_deb = shutil.which("dpkg-deb")
    if not dpkg_deb:
        raise RuntimeError("dpkg-deb is required to build the Debian package")

    _run([dpkg_deb, "--build", "--root-owner-group", str(package_root), str(package_path)])
    return package_path


if __name__ == "__main__":
    print(f"Built {build_package()}")
