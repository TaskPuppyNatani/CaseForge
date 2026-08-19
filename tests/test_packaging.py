"""Checks for reproducible PyInstaller and Debian packaging sources/layout."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmark_case_generator import __version__


ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_PACKAGE_FILES = (
    Path("opt/caseforge/CaseForge"),
    Path("opt/caseforge/_internal/base_library.zip"),
    Path("opt/caseforge/_internal/assets/icon.ico"),
    Path("opt/caseforge/_internal/assets/rivet_logo.png"),
    Path("usr/share/applications/caseforge.desktop"),
    Path("usr/share/icons/hicolor/256x256/apps/caseforge.png"),
)
_FORBIDDEN_PACKAGE_NAMES = {
    ".env",
    ".env.local",
    ".git",
    ".pytest_cache",
    ".venv",
    "api_key.txt",
    "credentials.json",
    "generator_state.json",
    "secrets.json",
    "tests",
}
_FORBIDDEN_ARCHIVE_SUFFIXES = {".7z", ".deb", ".gz", ".tar", ".tgz", ".zip"}


def _assert_debian_package_root(package_root: Path) -> None:
    """Assert the installed-tree contract without rebuilding or installing it."""

    missing = [
        relative
        for relative in _REQUIRED_PACKAGE_FILES
        if not (package_root / relative).is_file()
    ]
    assert not missing, f"missing packaged runtime files: {missing}"

    executable = package_root / "opt/caseforge/CaseForge"
    assert os.access(executable, os.X_OK), "packaged CaseForge is not executable"

    desktop = (package_root / "usr/share/applications/caseforge.desktop").read_text(
        encoding="utf-8"
    )
    assert "Exec=/opt/caseforge/CaseForge" in desktop
    assert "Terminal=false" in desktop
    assert "Icon=caseforge" in desktop

    top_level = {
        path.relative_to(package_root).parts[0]
        for path in package_root.iterdir()
    }
    assert top_level <= {"opt", "usr"}, f"unexpected package roots: {top_level}"

    forbidden = []
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        lower_parts = {part.lower() for part in relative.parts}
        if lower_parts & _FORBIDDEN_PACKAGE_NAMES:
            forbidden.append(relative)
            continue
        if any("workspace" in part or "qwen" in part for part in lower_parts):
            forbidden.append(relative)
            continue
        if (
            path.is_file()
            and path.name != "base_library.zip"
            and path.suffix.lower() in _FORBIDDEN_ARCHIVE_SUFFIXES
        ):
            forbidden.append(relative)

    assert not forbidden, f"forbidden development/package content: {forbidden}"
    assert not (package_root / "opt/caseforge/generator_state.json").exists()


def _make_deterministic_package_root(package_root: Path) -> Path:
    for relative in _REQUIRED_PACKAGE_FILES:
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"packaged runtime fixture")

    executable = package_root / "opt/caseforge/CaseForge"
    executable.chmod(0o755)
    (package_root / "usr/share/applications/caseforge.desktop").write_text(
        "\n".join(
            (
                "[Desktop Entry]",
                "Name=CaseForge",
                "Exec=/opt/caseforge/CaseForge",
                "Icon=caseforge",
                "Terminal=false",
                "Type=Application",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return package_root


def test_tracked_pyinstaller_spec_is_one_directory_and_windowed():
    spec = (ROOT / "CaseForge.spec").read_text(encoding="utf-8")

    assert 'name="CaseForge"' in spec
    assert "console=False" in spec
    assert '"assets"' in spec
    assert "COLLECT(" in spec


def test_desktop_entry_is_graphical_and_points_to_installed_executable():
    desktop = (ROOT / "debian" / "caseforge.desktop").read_text(encoding="utf-8")

    assert "Name=CaseForge" in desktop
    assert "Exec=/opt/caseforge/CaseForge" in desktop
    assert "Icon=caseforge" in desktop
    assert "Terminal=false" in desktop
    assert "Type=Application" in desktop


def test_debian_source_metadata_and_local_build_script_exist():
    control = (ROOT / "debian" / "control").read_text(encoding="utf-8")
    script = (ROOT / "packaging" / "build_deb.py").read_text(encoding="utf-8")

    assert "Source: caseforge" in control
    assert "Package: caseforge" in control
    assert "Architecture: amd64" in control
    assert not any(line.startswith("Version:") for line in control.splitlines())
    assert "PyInstaller" in script
    assert "dpkg-deb" in script
    assert "opt" in script and "caseforge" in script


def test_packaging_sources_are_not_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "*.spec" not in gitignore
    assert "debian/" not in gitignore
    assert "build/" in gitignore
    assert "dist/" in gitignore
    assert "*.deb" in gitignore


def test_shared_application_code_does_not_embed_linux_install_paths():
    shared_sources = [
        ROOT / "benchmark_case_generator" / "gui.py",
        ROOT / "benchmark_case_generator" / "resources.py",
    ]

    for source in shared_sources:
        assert "/opt/caseforge" not in source.read_text(encoding="utf-8")


def test_debian_package_layout_contract_on_deterministic_staging(tmp_path):
    package_root = _make_deterministic_package_root(tmp_path / "package-root")

    _assert_debian_package_root(package_root)


def test_existing_debian_package_contents_when_available(tmp_path):
    """Inspect the already-built package when packaging artifacts are present."""

    dpkg_deb = shutil.which("dpkg-deb")
    package = ROOT / "dist" / f"caseforge_{__version__}_amd64.deb"
    if dpkg_deb is None or not package.is_file():
        pytest.skip("dpkg-deb or the locally built CaseForge package is unavailable")

    package_root = tmp_path / "extracted-package"
    subprocess.run(
        [dpkg_deb, "--extract", str(package), str(package_root)],
        check=True,
    )

    _assert_debian_package_root(package_root)
