"""Focused regression tests for CaseForge table selection and Preview."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QItemSelectionModel, QSize
from PySide6.QtWidgets import QSizePolicy

from benchmark_case_generator import gui
from benchmark_case_generator.models import CasePlan, Difficulty, ExpectedConclusion, GeneratedCase
from benchmark_case_generator.theme import BRANDING_GREEN, BRANDING_PURPLE


def make_case(index: int) -> GeneratedCase:
    plan = CasePlan(
        title=f"Preview Case {index}",
        language="Python",
        technical_domain="validation",
        primary_concept=f"preview concept {index}",
        failure_mechanism="test mechanism",
        expected_conclusion=ExpectedConclusion.BUG,
        difficulty=Difficulty.MEDIUM,
        code_shape="function",
        semantic_signature=f"preview signature {index}",
        case_summary="A focused preview test case.",
    )
    return GeneratedCase(
        filename=f"{index:03d}_preview.md",
        markdown_content=f"# Preview {index}\n\nPrompt body {index}",
        markdown_sha256=f"sha-{index}",
        plan=plan,
        ground_truth_explanation="Private explanation",
        evidence_description="Private evidence",
        timestamp="2026-01-01T00:00:00Z",
        model_identifier="test-model",
    )


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"

    monkeypatch.setattr(gui, "settings_file", lambda: settings_path)
    monkeypatch.setattr(gui, "default_output_dir", lambda: tmp_path / "generated_tests")
    monkeypatch.setattr(gui, "default_state_file", lambda: tmp_path / "generator_state.json")

    def resolve_local(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(gui, "resolve_user_path", resolve_local)

    caseforge_window = gui.CaseForgeWindow()
    caseforge_window.cases_table.setRowCount(0)
    caseforge_window._generated_cases.clear()
    yield caseforge_window
    caseforge_window.close()
    caseforge_window.deleteLater()
    qapp.processEvents()


def add_cases(window, *cases):
    for case in cases:
        window._add_case_to_table(case)
        window._generated_cases.append(case)


def test_one_selected_row_resolves_to_one_case(window):
    case = make_case(1)
    add_cases(window, case)

    window.cases_table.selectRow(0)

    assert window._get_selected_cases() == [case]


def test_multiple_selected_cells_in_one_row_are_deduplicated(window):
    case = make_case(1)
    add_cases(window, case)
    selection_model = window.cases_table.selectionModel()

    for column in (0, 2, 6):
        selection_model.select(
            window.cases_table.model().index(0, column),
            QItemSelectionModel.Select,
        )

    assert window._get_selected_cases() == [case]


def test_multiple_selected_rows_resolve_in_table_order(window):
    cases = [make_case(1), make_case(2), make_case(3)]
    add_cases(window, *cases)
    selection_model = window.cases_table.selectionModel()

    for row in (2, 0):
        selection_model.select(
            window.cases_table.model().index(row, 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )

    assert window._get_selected_cases() == [cases[0], cases[2]]


def test_preview_opens_the_selected_case_without_mutating_it(window, monkeypatch):
    cases = [make_case(1), make_case(2)]
    add_cases(window, *cases)
    window.cases_table.selectRow(1)
    opened = []

    class FakePreviewDialog:
        def __init__(self, case, parent):
            opened.append(case)

        def exec(self):
            return 0

    monkeypatch.setattr(gui, "CasePreviewDialog", FakePreviewDialog)
    original_markdown = cases[1].markdown_content

    window._preview_selected()

    assert opened == [cases[1]]
    assert cases[1].markdown_content == original_markdown


def test_preview_without_selection_is_safe(window, monkeypatch):
    case = make_case(1)
    add_cases(window, case)
    opened = []

    class FakePreviewDialog:
        def __init__(self, selected_case, parent):
            opened.append(selected_case)

        def exec(self):
            return 0

    monkeypatch.setattr(gui, "CasePreviewDialog", FakePreviewDialog)
    window.cases_table.clearSelection()

    window._preview_selected()

    # Preserve the existing no-selection fallback while proving it no longer
    # reaches the singular Qt selection-API crash path.
    assert opened == [case]


def test_preview_with_no_cases_shows_graphical_message(window, monkeypatch):
    messages = []
    monkeypatch.setattr(
        gui.QMessageBox,
        "information",
        lambda *args: messages.append(args),
    )

    window._preview_selected()

    assert messages
    assert "No cases available to preview" in messages[0][-1]


def test_branding_is_in_bottom_action_row_and_colors_are_scoped(window):
    branding_index = window.action_button_layout.indexOf(window.branding_widget)
    delete_index = window.action_button_layout.indexOf(window.delete_btn)
    save_selected_index = window.action_button_layout.indexOf(window.save_selected_btn)

    assert delete_index < branding_index < save_selected_index
    assert BRANDING_GREEN in window.caseforge_header_label.styleSheet()
    assert BRANDING_PURPLE in window.branding_label.text()
    assert BRANDING_GREEN in window.branding_label.text()
    assert window.generated_cases_label.styleSheet() == ""
    assert window.branding_logo_label.pixmap() is not None
    assert not window.branding_logo_label.pixmap().isNull()
    assert window.branding_logo_label.pixmap().size() == QSize(44, 44)
    assert window.branding_logo_label.size() == QSize(46, 46)
    assert window.branding_label.font().pointSize() == 13
    assert window.branding_label.font().bold()
    assert window.branding_widget.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert window.action_button_layout.stretch(branding_index) == 1
    assert window.branding_widget.layout().spacing() == 10
    assert window.branding_widget.layout().contentsMargins().left() == 12
    assert window.branding_widget.layout().contentsMargins().right() == 12


def test_gui_settings_save_excludes_api_key_but_persists_ordinary_settings(
    window, monkeypatch, tmp_path
):
    fake_api_key = "caseforge-test-only-fake-api-key"

    class FakeSignal:
        def connect(self, _slot):
            pass

    class FakeWorker:
        def __init__(self, **kwargs):
            self.progress = FakeSignal()
            self.case_generated = FakeSignal()
            self.generation_complete = FakeSignal()
            self.error_occurred = FakeSignal()
            self.client_config = kwargs["client_config"]

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(gui, "GenerationWorker", FakeWorker)
    window.api_key_input.setText(fake_api_key)
    window.api_key_env_input.setText("CASEFORGE_TEST_API_KEY")
    window.base_url_input.setText("https://example.test/v1")
    window.model_input.setText("coverage-model")
    window.output_dir_input.setText(str(tmp_path / "saved-cases"))

    window._start_generation()

    settings_path = gui.settings_file()
    persisted_text = settings_path.read_text(encoding="utf-8")
    persisted = json.loads(persisted_text)

    assert persisted["api_key_env"] == "CASEFORGE_TEST_API_KEY"
    assert persisted["base_url"] == "https://example.test/v1"
    assert persisted["model"] == "coverage-model"
    assert "api_key" not in persisted
    assert persisted_text.find(fake_api_key) == -1, "fake API key was persisted"
    if window.api_key_input.text() != fake_api_key:
        pytest.fail("live widget should retain the API key for the current session")

    reloaded = gui.CaseForgeWindow()
    try:
        assert reloaded.api_key_input.text() == ""
        assert reloaded.model_input.text() == "coverage-model"
        assert reloaded.base_url_input.text() == "https://example.test/v1"
    finally:
        reloaded.close()
        reloaded.deleteLater()
