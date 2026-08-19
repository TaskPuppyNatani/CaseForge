"""Shared Qt test setup for headless CaseForge GUI tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application
    if QApplication.instance() is application:
        application.quit()
