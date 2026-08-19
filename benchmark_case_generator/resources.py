"""Cross-platform application resources and user-writable paths."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QStandardPaths
from PySide6.QtGui import QIcon


APP_NAME = "CaseForge"
APP_ORGANIZATION = "CaseForge"


def resource_root() -> Path:
    """Return the directory containing bundled runtime resources.

    Source runs resolve from the package location, while PyInstaller runs
    resolve from its temporary/embedded bundle directory. Keeping that
    decision here prevents GUI code from depending on the current directory
    or on PyInstaller-specific globals.
    """

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: str | Path) -> Optional[Path]:
    """Resolve an optional bundled resource independently of the CWD."""

    candidate = resource_root() / Path(relative_path)
    return candidate if candidate.is_file() else None


def user_data_dir() -> Path:
    """Return the platform-appropriate writable application-data directory."""

    for location in (
        QStandardPaths.writableLocation(QStandardPaths.AppDataLocation),
        QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation),
    ):
        if location:
            return Path(location)

    # QStandardPaths normally always supplies a location. This fallback is
    # deliberately platform-neutral and is only for unusual test/runtime
    # environments where Qt cannot determine one.
    return Path.home() / APP_NAME


def user_data_file(filename: str | Path) -> Path:
    """Return a file path under the writable application-data directory."""

    return user_data_dir() / Path(filename)


def settings_file() -> Path:
    """Return the settings file path without creating it."""

    return user_data_file("settings.json")


def default_state_file() -> Path:
    """Return the private generator-state path without creating it."""

    return user_data_file("generator_state.json")


def default_output_dir() -> Path:
    """Return the default user-selected output directory."""

    return user_data_dir() / "generated_tests"


def resolve_user_path(value: str | Path) -> Path:
    """Resolve a configured path without making relative paths CWD-dependent."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else user_data_dir() / path


def load_application_icon() -> QIcon:
    """Load the optional CaseForge icon, returning an empty icon if absent."""

    icon_path = resource_path("assets/icon.ico")
    return QIcon(str(icon_path)) if icon_path else QIcon()
