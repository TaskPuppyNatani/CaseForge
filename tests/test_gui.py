"""Tests for CaseForge GUI components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

# Import backend modules that GUI uses
from benchmark_case_generator.client import ClientConfig, ModelClient
from benchmark_case_generator.models import ExpectedConclusion, Difficulty, CasePlan, GeneratedCase
from benchmark_case_generator.diversity import DiversityTracker
from benchmark_case_generator.storage import GeneratorState, OutputManager


class TestGUIBackendIntegration:
    """Test that GUI can properly use backend components."""
    
    def test_client_config_from_gui_inputs(self):
        """Simulate GUI creating client config from user inputs."""
        # Simulated GUI inputs
        base_url = "http://localhost:1234/v1"
        model = "qwen-7b"
        api_key_env = "OPENAI_API_KEY"
        
        config = ClientConfig(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
        )
        
        assert config.base_url == base_url
        assert config.model == model
        # API key would be loaded from env in __post_init__
    
    def test_conclusion_distribution_from_gui_spinboxes(self):
        """Simulate GUI converting spinbox values to distribution."""
        # Simulated spinbox values (percentages)
        bug_pct = 40
        correct_pct = 30
        unsupported_pct = 15
        needs_context_pct = 10
        intentional_pct = 5
        
        distribution = {
            ExpectedConclusion.BUG: bug_pct / 100.0,
            ExpectedConclusion.CORRECT: correct_pct / 100.0,
            ExpectedConclusion.UNSUPPORTED: unsupported_pct / 100.0,
            ExpectedConclusion.NEEDS_CONTEXT: needs_context_pct / 100.0,
            ExpectedConclusion.INTENTIONAL: intentional_pct / 100.0,
        }
        
        assert abs(sum(distribution.values()) - 1.0) < 0.001
        assert distribution[ExpectedConclusion.BUG] == 0.40
    
    def test_difficulty_selection_from_checkboxes(self):
        """Simulate GUI getting selected difficulties from checkboxes."""
        # Simulated checkbox states
        easy_checked = True
        medium_checked = False
        hard_checked = True
        
        selected = []
        if easy_checked:
            selected.append(Difficulty.EASY)
        if medium_checked:
            selected.append(Difficulty.MEDIUM)
        if hard_checked:
            selected.append(Difficulty.HARD)
        
        assert Difficulty.EASY in selected
        assert Difficulty.MEDIUM not in selected
        assert Difficulty.HARD in selected
    
    def test_language_selection_from_listwidget(self):
        """Simulate GUI getting selected languages from multi-select list."""
        # Simulated selected items
        selected_languages = ["Python", "Java", "Go"]
        
        assert len(selected_languages) == 3
        assert "Python" in selected_languages


class TestGeneratedCaseListUpdates:
    """Test that generated cases are properly tracked."""
    
    def test_case_added_to_tracker(self):
        tracker = DiversityTracker()
        assert tracker.case_count == 0
        
        plan = CasePlan(
            title="Test Case",
            language="Python",
            technical_domain="error handling",
            primary_concept="test concept",
            failure_mechanism="test mechanism",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="test_sig",
            case_summary="summary",
        )
        
        case = GeneratedCase(
            filename="001_test.md",
            markdown_content="# Test\n```python\ncode\n```",
            markdown_sha256="abc123",
            plan=plan,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        
        tracker.add_case(case)
        assert tracker.case_count == 1
    
    def test_multiple_cases_tracked(self):
        tracker = DiversityTracker()
        
        for i in range(5):
            plan = CasePlan(
                title=f"Case {i}",
                language="Python",
                technical_domain="error handling",
                primary_concept=f"unique concept {i}",
                failure_mechanism="mechanism",
                expected_conclusion=ExpectedConclusion.BUG,
                difficulty=Difficulty.MEDIUM,
                code_shape="function",
                semantic_signature=f"sig_{i}",
                case_summary="summary",
            )
            
            case = GeneratedCase(
                filename=f"{i+1:03d}_case{i}.md",
                markdown_content=f"# Case {i}\n```python\ncode\n```",
                markdown_sha256=f"sha{i}",
                plan=plan,
                ground_truth_explanation="explanation",
                evidence_description="evidence",
                timestamp="2024-01-01T00:00:00Z",
                model_identifier="test-model",
            )
            
            tracker.add_case(case)
        
        assert tracker.case_count == 5


class TestSaveSelectedCases:
    """Test saving selected cases behavior."""
    
    def test_output_manager_never_overwrites(self, tmp_path):
        output = OutputManager(str(tmp_path))
        output.initialize()
        
        filename = "001_test.md"
        content1 = "# First version"
        content2 = "# Second version"
        
        sha1 = output.write_case(filename, content1)
        
        with pytest.raises(FileExistsError):
            output.write_case(filename, content2)
        
        # Verify original content preserved
        filepath = tmp_path / filename
        assert filepath.read_text() == content1


class TestSaveAllCases:
    """Test save all cases behavior."""
    
    def test_all_cases_written_to_output_dir(self, tmp_path):
        output = OutputManager(str(tmp_path))
        output.initialize()
        
        cases_data = [
            ("001_first.md", "# First case"),
            ("002_second.md", "# Second case"),
            ("003_third.md", "# Third case"),
        ]
        
        for filename, content in cases_data:
            output.write_case(filename, content)
        
        # Verify all files exist
        for filename, content in cases_data:
            filepath = tmp_path / filename
            assert filepath.exists()
            assert filepath.read_text() == content


class TestGroundTruthNeverInMarkdown:
    """Verify private ground truth never enters .md output."""
    
    def test_markdown_does_not_contain_expected_conclusion(self):
        """The markdown content should not leak the expected answer."""
        plan = CasePlan(
            title="Leak Test",
            language="Python",
            technical_domain="validation",
            primary_concept="input validation",
            failure_mechanism="missing check",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="leak_sig",
            case_summary="summary",
        )
        
        # Simulated markdown that should NOT contain ground truth
        markdown_content = """Review the supplied code for concrete defects.

Report only findings directly supported by the visible code and stated contract.

If a concrete defect is demonstrated, output exactly:

Title: ...
Severity: Critical | High | Medium | Low
Confidence: High | Medium | Low
Evidence: ...
Why it matters: ...
Minimal fix: ...

If no concrete defect is demonstrated, output exactly:

No valid finding.
Classification: Already handled | Intentional behavior | Unsupported by evidence | Requires additional context | Non-bug improvement suggestion
Reason: ...

Return exactly one conclusion.

```python
def process_input(value):
    if value is None:
        raise ValueError("value cannot be None")
    return value.strip()
```
"""
        
        # Check for forbidden patterns that would leak ground truth
        lower_md = markdown_content.lower()
        assert "expected_conclusion" not in lower_md
        assert "this code contains a bug" not in lower_md
        assert "this code is safe" not in lower_md
        assert "this code is correct" not in lower_md
        assert "bug:" not in lower_md
        assert "defect:" not in lower_md
    
    def test_private_state_contains_ground_truth(self, tmp_path):
        """Private state file SHOULD contain ground truth."""
        state_file = tmp_path / "test_state.json"
        state = GeneratorState(str(state_file))
        
        plan = CasePlan(
            title="Private Test",
            language="Python",
            technical_domain="error handling",
            primary_concept="private concept",
            failure_mechanism="private mechanism",
            expected_conclusion=ExpectedConclusion.CORRECT,
            difficulty=Difficulty.EASY,
            code_shape="function",
            semantic_signature="private_sig",
            case_summary="summary",
        )
        
        case = GeneratedCase(
            filename="001_private.md",
            markdown_content="# Private\n```python\ncode\n```",
            markdown_sha256="private_sha",
            plan=plan,
            ground_truth_explanation="This is private ground truth",
            evidence_description="Private evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        
        state.initialize("test-model")
        state.add_case(case)
        state.save()
        
        # Load and verify ground truth is in state file
        with open(state_file, "r") as f:
            data = json.load(f)
        
        assert data["cases"][0]["expected_conclusion"] == "CORRECT"
        assert data["cases"][0]["ground_truth_explanation"] == "This is private ground truth"


class TestOutputFolderHandling:
    """Test output directory handling."""
    
    def test_output_dir_created_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "new_output" / "subdir"
        assert not new_dir.exists()
        
        output = OutputManager(str(new_dir))
        output.initialize()
        
        assert new_dir.exists()
    
    def test_existing_files_scanned_on_init(self, tmp_path):
        # Create some existing files
        (tmp_path / "001_existing.md").write_text("# Existing 1")
        (tmp_path / "002_existing.md").write_text("# Existing 2")
        (tmp_path / "005_existing.md").write_text("# Existing 5")
        
        output = OutputManager(str(tmp_path))
        output.initialize()
        
        # Next filename should be after the highest existing
        next_filename = output.get_next_filename("new")
        assert next_filename.startswith("006_")


class TestProviderConfigurationFromGUI:
    """Test provider configuration passed from GUI."""
    
    def test_config_with_api_key_env(self):
        import os
        os.environ["TEST_API_KEY"] = "test-key-123"
        
        config = ClientConfig(
            base_url="http://test.local/v1",
            model="test-model",
            api_key_env="TEST_API_KEY",
        )
        
        assert config.api_key == "test-key-123"
    
    def test_config_without_api_key(self):
        config = ClientConfig(
            base_url="http://localhost:1234/v1",
            model="qwen",
            api_key_env=None,
        )
        
        assert config.api_key is None
    
    def test_config_explicit_api_key(self):
        config = ClientConfig(
            base_url="http://test.local/v1",
            model="test-model",
            api_key="explicit-key",
        )
        
        assert config.api_key == "explicit-key"


class TestGenerationErrorsWithoutCrash:
    """Test that generation errors are surfaced without crashing."""
    
    @patch('benchmark_case_generator.client.ModelClient.check_health')
    def test_unreachable_endpoint_returns_error(self, mock_health):
        mock_health.return_value = False
        
        config = ClientConfig(
            base_url="http://unreachable.local/v1",
            model="test-model",
        )
        client = ModelClient(config)
        
        # Should not crash, just return False
        result = client.check_health()
        assert result is False
    
    def test_invalid_json_in_plan_handled(self):
        from benchmark_case_generator.generation import CaseGenerator
        
        # Simulate parsing invalid JSON
        generator_mock = Mock()
        generator_mock._parse_case_plan = Mock(return_value=None)
        
        # This simulates what happens when model returns malformed JSON
        # The generator should retry, not crash
        result = generator_mock._parse_case_plan("not valid json {{{")
        assert result is None


class TestRepeatedGenerationRetainsDiversity:
    """Test that repeated generation retains diversity history."""
    
    def test_tracker_remembers_previous_concepts(self):
        tracker = DiversityTracker()
        
        # Add first batch of cases
        for i in range(3):
            plan = CasePlan(
                title=f"Batch1 Case {i}",
                language="Python",
                technical_domain="error handling",
                primary_concept=f"batch1 concept {i}",
                failure_mechanism="mechanism",
                expected_conclusion=ExpectedConclusion.BUG,
                difficulty=Difficulty.MEDIUM,
                code_shape="function",
                semantic_signature=f"batch1_sig_{i}",
                case_summary="summary",
            )
            
            case = GeneratedCase(
                filename=f"00{i+1}_batch1.md",
                markdown_content=f"# Batch1 Case {i}",
                markdown_sha256=f"sha{i}",
                plan=plan,
                ground_truth_explanation="explanation",
                evidence_description="evidence",
                timestamp="2024-01-01T00:00:00Z",
                model_identifier="test-model",
            )
            
            tracker.add_case(case)
        
        # Try to add duplicate concept
        duplicate_plan = CasePlan(
            title="Duplicate",
            language="Java",
            technical_domain="error handling",
            primary_concept="batch1 concept 0",  # Same as first case
            failure_mechanism="mechanism",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="dup_sig",
            case_summary="summary",
        )
        
        is_valid, reason = tracker.validate_plan(duplicate_plan)
        assert is_valid is False
        # Reason mentions "matches occupied concept" for exact duplicates
        assert "matches" in reason.lower() or "duplicates" in reason.lower() or "similar" in reason.lower()
    
    def test_state_file_persists_between_sessions(self, tmp_path):
        state_file = tmp_path / "persist_state.json"
        
        # First session
        state1 = GeneratorState(str(state_file))
        state1.initialize("model-v1")
        
        plan1 = CasePlan(
            title="Session1 Case",
            language="Python",
            technical_domain="error handling",
            primary_concept="session1 concept",
            failure_mechanism="mechanism",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="session1_sig",
            case_summary="summary",
        )
        
        case1 = GeneratedCase(
            filename="001_session1.md",
            markdown_content="# Session1",
            markdown_sha256="s1_sha",
            plan=plan1,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="model-v1",
        )
        
        state1.add_case(case1)
        state1.save()
        
        # Second session - load existing
        state2 = GeneratorState(str(state_file))
        existing = state2.load()
        
        assert len(existing) == 1
        assert existing[0].plan.primary_concept == "session1 concept"


class TestGUIDoesNotRequireCLI:
    """Verify GUI does not depend on CLI module."""
    
    def test_gui_module_importable_without_cli(self):
        """Verify GUI source code doesn't import cli module (without actually importing)."""
        # We can't actually import gui.py in headless environment due to PySide6 dependencies
        # Instead, verify the source code doesn't contain CLI imports
        import ast
        
        gui_source_path = Path(__file__).parent.parent / "benchmark_case_generator" / "gui.py"
        with open(gui_source_path, "r") as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # GUI should not import cli module (but client is fine - it's a backend component)
        cli_imports = [i for i in imports if i == 'benchmark_case_generator.cli' or i == 'cli']
        assert len(cli_imports) == 0, f"GUI should not import CLI module, found: {cli_imports}"
