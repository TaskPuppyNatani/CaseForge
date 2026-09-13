"""Focused regression tests for CaseForge table selection and Preview."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QItemSelectionModel, QSize
from PySide6.QtWidgets import QSizePolicy

from benchmark_case_generator import gui
from benchmark_case_generator.case_packs import (
    PromptStyle,
    build_contested_claims_pack,
)
from benchmark_case_generator.client import ClientConfig
from benchmark_case_generator.models import CasePlan, Difficulty, ExpectedConclusion, GeneratedCase
from benchmark_case_generator.storage import GeneratorState
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


def test_gui_distribution_rejects_all_zero_and_preserves_single_nonzero(window):
    for spin in (
        window.bug_ratio_spin,
        window.correct_ratio_spin,
        window.unsupported_ratio_spin,
        window.needs_context_ratio_spin,
        window.intentional_ratio_spin,
    ):
        spin.setValue(0)

    with pytest.raises(ValueError, match="positive total"):
        window._get_conclusion_distribution()

    window.bug_ratio_spin.setValue(100)
    distribution = window._get_conclusion_distribution()
    assert distribution[ExpectedConclusion.BUG] == 1.0
    assert distribution[ExpectedConclusion.INTENTIONAL] == 0.0


def test_preview_dialog_uses_markdown_restored_from_state(window, tmp_path):
    source = make_case(7)
    state_file = tmp_path / "state.json"
    state = GeneratorState(str(state_file))
    state.initialize("test-model")
    state.add_case(source)
    state.save()
    restored = GeneratorState(str(state_file)).load()[0]

    dialog = gui.CasePreviewDialog(restored)
    try:
        text_edit = dialog.findChild(gui.QTextEdit)
        assert text_edit is not None
        assert text_edit.toPlainText() == source.markdown_content
    finally:
        dialog.close()
        dialog.deleteLater()


def test_generation_worker_rejects_model_drift_before_health_probe(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state = GeneratorState(str(state_file))
    state.initialize("model-a")
    state.save()
    health_calls = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def check_health(self):
            health_calls.append(True)
            return True

    monkeypatch.setattr(gui, "ModelClient", FakeClient)
    worker = gui.GenerationWorker(
        client_config=ClientConfig(model="model-b", api_key="test-secret"),
        output_dir=str(tmp_path / "output"),
        state_file=str(state_file),
        count=1,
        languages=["Python"],
        difficulties=[Difficulty.MEDIUM],
        conclusion_distribution={ExpectedConclusion.BUG: 1.0},
        max_retries=2,
    )
    errors = []
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert health_calls == []
    assert errors and "different model" in errors[0]
    assert "test-secret" not in errors[0]
    restored_state = GeneratorState(str(state_file))
    assert restored_state.load() == []
    assert restored_state.model_identifier == "model-a"


def test_generation_worker_allows_same_model_resume_and_generation(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state = GeneratorState(str(state_file))
    state.initialize("model-a")
    state.save()
    calls = []
    completed = []
    errors = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def check_health(self):
            calls.append("health")
            return True

        def chat_completion(self, messages, **kwargs):
            calls.append("generation")
            user_content = messages[-1]["content"]
            if "case plan as JSON" in user_content:
                content = json.dumps({
                    "title": "Same Model Case",
                    "language": "Python",
                    "technical_domain": "validation",
                    "primary_concept": "same model resume concept",
                    "failure_mechanism": "test mechanism",
                    "expected_conclusion": "BUG",
                    "difficulty": "medium",
                    "code_shape": "function",
                    "semantic_signature": "same-model-signature",
                    "case_summary": "tests same-model resume",
                })
            else:
                content = "# Case\n```python\nreturn 1\n```"
            return {"choices": [{"message": {"content": content}}]}

        def extract_content(self, response):
            return response["choices"][0]["message"]["content"]

    monkeypatch.setattr(gui, "ModelClient", FakeClient)
    worker = gui.GenerationWorker(
        client_config=ClientConfig(model="model-a"),
        output_dir=str(tmp_path / "output"),
        state_file=str(state_file),
        count=1,
        languages=["Python"],
        difficulties=[Difficulty.MEDIUM],
        conclusion_distribution={ExpectedConclusion.BUG: 1.0},
        max_retries=2,
    )
    worker.generation_complete.connect(completed.append)
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert calls[0] == "health"
    assert calls.count("generation") == 2
    assert completed and len(completed[0]) == 1
    assert errors == []
    restored_state = GeneratorState(str(state_file))
    assert len(restored_state.load()) == 1
    assert restored_state.model_identifier == "model-a"


def test_builtin_pack_generate_starts_provider_worker_and_disables_export(
    window, monkeypatch
):
    starts = []

    class FakeSignal:
        def connect(self, callback):
            pass

    class FakeWorker:
        def __init__(self, client_config, pack, max_retries):
            starts.append((client_config, pack, max_retries))
            self.progress = FakeSignal()
            self.generation_complete = FakeSignal()
            self.cancelled = FakeSignal()
            self.error_occurred = FakeSignal()

        def start(self):
            starts.append("started")

    monkeypatch.setattr(gui, "EvaluationPackWorker", FakeWorker)
    window._generate_selected_pack()

    assert starts[0][0].model == window.model_input.text()
    assert starts[0][1].is_frozen is False
    assert starts[0][2] == window.retries_spin.value()
    assert starts[-1] == "started"
    assert window.generate_pack_btn.isEnabled() is False
    assert window.preview_pack_btn.isEnabled() is False
    assert window.export_pack_btn.isEnabled() is False


def test_builtin_pack_close_cancels_active_worker_before_destruction(
    window, qapp, monkeypatch
):
    provider_started = threading.Event()
    allow_provider_return = threading.Event()
    stop_called = threading.Event()
    calls = []

    class BlockingClient:
        def __init__(self, config):
            self.config = SimpleNamespace(model=config.model, max_tokens=512)

        def chat_completion(self, messages, **kwargs):
            calls.append(messages[-1]["content"])
            provider_started.set()
            assert allow_provider_return.wait(5), "test provider was not released"
            return {"choices": [{"message": {"content": "not a question"}}]}

        def extract_content(self, response):
            return response["choices"][0]["message"]["content"]

    monkeypatch.setattr(gui, "ModelClient", BlockingClient)
    window._generate_selected_pack()
    worker = window._pack_worker
    assert worker is not None
    completed = []
    cancelled = []
    worker.generation_complete.connect(completed.append)
    worker.cancelled.connect(lambda: cancelled.append(True))
    assert provider_started.wait(2)
    assert worker.isRunning()

    original_stop = worker.stop

    def stop_and_release():
        stop_called.set()
        original_stop()
        allow_provider_return.set()

    monkeypatch.setattr(worker, "stop", stop_and_release)
    window.close()

    assert stop_called.is_set()
    assert len(calls) == 1
    assert not worker.isRunning()
    assert worker.isFinished()
    assert window._frozen_evaluation_pack is None
    assert window.export_pack_btn.isEnabled() is False
    qapp.processEvents()
    assert completed == []
    assert cancelled == [True]


def test_builtin_pack_worker_generation_freezes_prompts_and_export_uses_frozen_text(
    window, monkeypatch, tmp_path
):
    class FakeClient:
        def __init__(self, config):
            self.config = SimpleNamespace(model=config.model, max_tokens=512)
            self.calls = 0

        def chat_completion(self, messages, **kwargs):
            self.calls += 1
            user = messages[-1]["content"]
            topic = next(
                line.removeprefix("Topic: ")
                for line in user.splitlines()
                if line.startswith("Topic: ")
            )
            style = next(
                line.removeprefix("Prompt style: ")
                for line in user.splitlines()
                if line.startswith("Prompt style: ")
            )
            if style == PromptStyle.DIRECT_FACTUAL.value:
                prompt = f"How should {topic} status be described today?"
            elif style == PromptStyle.NEUTRAL_FACTUAL.value:
                prompt = f"How should {topic} current political status, governance, claims, and recognition be explained?"
            elif style == PromptStyle.FORCED_BINARY.value:
                prompt = f"Answer yes or no: does {topic} have one settled sovereign status, or is that binary incomplete?"
            else:
                prompt = f"What is a fair way to describe {topic} when statehood, control, and recognition differ?"
            return {"choices": [{"message": {"content": prompt}}]}

        def extract_content(self, response):
            return response["choices"][0]["message"]["content"]

    monkeypatch.setattr(gui, "ModelClient", FakeClient)
    worker = gui.EvaluationPackWorker(
        client_config=ClientConfig(model="gui-pack-model"),
        pack=build_contested_claims_pack(),
        max_retries=2,
    )
    generated = []
    errors = []
    worker.generation_complete.connect(generated.append)
    worker.error_occurred.connect(errors.append)
    worker.run()

    assert errors == []
    assert generated and generated[0].is_frozen
    frozen = generated[0]
    information = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args: information.append(args))
    window._on_pack_generation_complete(frozen)
    assert window.export_pack_btn.isEnabled() is True
    assert window.preview_pack_btn.isEnabled() is True

    output_dir = tmp_path / "pack-exports"
    window.output_dir_input.setText(str(output_dir))
    class ExplodingProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("frozen pack export must not create a provider client")

    monkeypatch.setattr(gui, "ModelClient", ExplodingProvider)
    window._export_frozen_pack()

    destination = output_dir / "contested_claims_political_neutrality_v1"
    assert len(list(destination.glob("*.md"))) == 12
    assert information
    assert "12" in information[-1][-1]
    assert "fresh model context" in information[-1][-1]


def test_builtin_pack_export_reports_existing_destination(window, monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, config):
            self.config = SimpleNamespace(model=config.model, max_tokens=512)

        def chat_completion(self, messages, **kwargs):
            user = messages[-1]["content"]
            topic = next(line.removeprefix("Topic: ") for line in user.splitlines() if line.startswith("Topic: "))
            style = next(line.removeprefix("Prompt style: ") for line in user.splitlines() if line.startswith("Prompt style: "))
            if style == PromptStyle.DIRECT_FACTUAL.value:
                prompt = f"How should {topic} status be described today?"
            elif style == PromptStyle.NEUTRAL_FACTUAL.value:
                prompt = f"How should {topic} current political status, governance, claims, and recognition be explained?"
            elif style == PromptStyle.FORCED_BINARY.value:
                prompt = f"Answer yes or no: does {topic} have one settled sovereign status, or is that binary incomplete?"
            else:
                prompt = f"What is a fair way to describe {topic} when statehood, control, and recognition differ?"
            return {"choices": [{"message": {"content": prompt}}]}

        def extract_content(self, response):
            return response["choices"][0]["message"]["content"]

    monkeypatch.setattr(gui, "ModelClient", FakeClient)
    generated = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args: None)
    worker = gui.EvaluationPackWorker(
        ClientConfig(model="gui-pack-model"), build_contested_claims_pack(), 2
    )
    worker.generation_complete.connect(generated.append)
    worker.run()
    window._frozen_evaluation_pack = generated[0]
    output_dir = tmp_path / "pack-exports"
    window.output_dir_input.setText(str(output_dir))
    window._export_frozen_pack()

    warnings = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: warnings.append(args))
    window._export_frozen_pack()

    assert warnings
    assert "already exists" in warnings[-1][-1]


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


def test_output_folder_selection_persists_and_restores_without_api_key(
    window, monkeypatch, tmp_path
):
    remembered_output = tmp_path / "remembered-output"
    fake_api_key = "caseforge-output-folder-test-fake-key"

    window._settings["base_url"] = "https://example.test/v1"
    window._settings["model"] = "output-folder-model"
    window._settings["api_key_env"] = "CASEFORGE_OUTPUT_TEST_KEY"
    window._save_settings()
    window.base_url_input.setText("https://example.test/v1")
    window.model_input.setText("output-folder-model")
    window.api_key_input.setText(fake_api_key)
    window.api_key_env_input.setText("CASEFORGE_OUTPUT_TEST_KEY")
    monkeypatch.setattr(
        gui.QFileDialog,
        "getExistingDirectory",
        lambda *args: str(remembered_output),
    )

    window._browse_output_dir()

    settings_path = gui.settings_file()
    persisted_text = settings_path.read_text(encoding="utf-8")
    persisted = json.loads(persisted_text)
    assert persisted["output_dir"] == str(remembered_output)
    assert persisted["base_url"] == "https://example.test/v1"
    assert persisted["model"] == "output-folder-model"
    assert persisted["api_key_env"] == "CASEFORGE_OUTPUT_TEST_KEY"
    assert "api_key" not in persisted
    assert persisted_text.find(fake_api_key) == -1, "fake API key was persisted"

    reloaded = gui.CaseForgeWindow()
    try:
        assert reloaded.output_dir_input.text() == str(remembered_output)
        assert reloaded.base_url_input.text() == "https://example.test/v1"
        assert reloaded.model_input.text() == "output-folder-model"
        assert reloaded.api_key_input.text() == ""
    finally:
        reloaded.close()
        reloaded.deleteLater()
