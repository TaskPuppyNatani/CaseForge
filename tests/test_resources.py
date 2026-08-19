"""Tests for source/frozen resource resolution and user-state paths."""

from __future__ import annotations

from pathlib import Path

from benchmark_case_generator import resources


def test_resource_resolution_does_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    icon_path = resources.resource_path("assets/icon.ico")

    assert icon_path is not None
    assert icon_path.is_file()
    assert icon_path.name == "icon.ico"


def test_missing_optional_resource_fails_gracefully():
    assert resources.resource_path("assets/does-not-exist.png") is None


def test_frozen_resource_root_uses_bundle_directory(monkeypatch, tmp_path):
    asset = tmp_path / "assets" / "icon.ico"
    asset.parent.mkdir()
    asset.write_bytes(b"test")
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)

    assert resources.resource_path("assets/icon.ico") == asset


def test_application_icon_is_available_when_intended_asset_exists(qapp):
    icon = resources.load_application_icon()

    assert not icon.isNull()


def test_user_state_path_is_not_inside_linux_install_prefix():
    path = resources.default_state_file()

    assert not path.is_relative_to(Path("/opt/caseforge"))
