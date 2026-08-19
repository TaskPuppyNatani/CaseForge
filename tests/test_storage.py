"""Tests for storage management."""

import pytest
import json
import hashlib
import tempfile
from pathlib import Path

from benchmark_case_generator.storage import GeneratorState, OutputManager
from benchmark_case_generator.models import CasePlan, GeneratedCase, ExpectedConclusion, Difficulty


class TestGeneratorState:
    def test_load_nonexistent(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name
        
        # Remove the file so it doesn't exist
        Path(state_file).unlink()
        
        state = GeneratorState(state_file)
        cases = state.load()
        
        assert cases == []
        assert state.case_count == 0
    
    def test_load_existing(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            data = {
                "version": "1.0",
                "generated_at": "2024-01-01T00:00:00Z",
                "model_identifier": "test-model",
                "cases": [
                    {
                        "filename": "001_test.md",
                        "markdown_sha256": "abc123",
                        "title": "Test Case",
                        "language": "Python",
                        "technical_domain": "error handling",
                        "primary_concept": "test concept",
                        "failure_mechanism": "test mechanism",
                        "expected_conclusion": "BUG",
                        "difficulty": "medium",
                        "semantic_signature": "sig",
                        "ground_truth_explanation": "explanation",
                        "evidence_description": "evidence",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "model_identifier": "test-model",
                    }
                ]
            }
            json.dump(data, f)
            state_file = f.name
        
        state = GeneratorState(state_file)
        cases = state.load()
        
        assert len(cases) == 1
        assert cases[0].filename == "001_test.md"
        assert state.case_count == 1
        assert state.model_identifier == "test-model"
    
    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            
            state = GeneratorState(str(state_file))
            state.initialize("test-model")
            
            plan = CasePlan(
                title="Save Test",
                language="Java",
                technical_domain="validation",
                primary_concept="save concept",
                failure_mechanism="validation error",
                expected_conclusion=ExpectedConclusion.CORRECT,
                difficulty=Difficulty.EASY,
                code_shape="class",
                semantic_signature="save_sig",
                case_summary="summary",
            )
            
            case = GeneratedCase(
                filename="001_save.md",
                markdown_content="# Save Test",
                markdown_sha256="save123",
                plan=plan,
                ground_truth_explanation="explanation",
                evidence_description="evidence",
                timestamp="2024-01-01T00:00:00Z",
                model_identifier="test-model",
            )
            
            state.add_case(case)
            state.save()
            
            # Verify file was created
            assert state_file.exists()
            
            # Verify content
            with open(state_file) as f:
                data = json.load(f)
            
            assert data["model_identifier"] == "test-model"
            assert len(data["cases"]) == 1
            assert data["cases"][0]["filename"] == "001_save.md"
    
    def test_initialize_sets_timestamp(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name
        
        Path(state_file).unlink()
        
        state = GeneratorState(state_file)
        state.initialize("test-model")
        
        assert state._data["generated_at"] is not None
        assert state._data["model_identifier"] == "test-model"


class TestOutputManager:
    def test_initialize_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            
            manager = OutputManager(str(output_dir))
            manager.initialize()
            
            assert output_dir.exists()
    
    def test_get_next_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            filename = manager.get_next_filename("test case")
            
            assert filename.startswith("001_")
            assert filename.endswith(".md")
            assert "test_case" in filename.lower()
    
    def test_write_case_returns_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            content = "# Test\n```python\ncode\n```"
            sha256 = manager.write_case("001_test.md", content)
            
            # Verify SHA-256 is correct
            expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
            assert sha256 == expected
            
            # Verify file was written
            filepath = Path(tmpdir) / "001_test.md"
            assert filepath.exists()
            with open(filepath) as f:
                assert f.read() == content
    
    def test_write_case_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            content1 = "# First"
            manager.write_case("001_test.md", content1)
            
            content2 = "# Second"
            with pytest.raises(FileExistsError):
                manager.write_case("001_test.md", content2)
    
    def test_file_exists_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            assert manager.file_exists("nonexistent.md") is False
            
            manager.write_case("001_exists.md", "# Exists")
            
            assert manager.file_exists("001_exists.md") is True
    
    def test_numbering_continues_after_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing files
            Path(tmpdir).joinpath("001_first.md").write_text("# First")
            Path(tmpdir).joinpath("002_second.md").write_text("# Second")
            
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            # Next should be 003
            filename = manager.get_next_filename("third")
            assert filename.startswith("003_")
    
    def test_sanitizes_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            manager.initialize()
            
            # Filename with special chars
            filename = manager.get_next_filename("Test@Case#With$Special%Chars!")
            
            # Should only contain alphanumeric, dash, underscore
            stem = filename.replace(".md", "").replace("001_", "")
            assert all(c.isalnum() or c == '_' for c in stem)
