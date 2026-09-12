"""Tests for models and data serialization."""

import pytest
import json
from datetime import datetime

from benchmark_case_generator.models import (
    CasePlan, GeneratedCase, ExpectedConclusion, Difficulty
)


class TestExpectedConclusion:
    def test_enum_values(self):
        assert ExpectedConclusion.BUG.value == "BUG"
        assert ExpectedConclusion.CORRECT.value == "CORRECT"
        assert ExpectedConclusion.UNSUPPORTED.value == "UNSUPPORTED"
        assert ExpectedConclusion.NEEDS_CONTEXT.value == "NEEDS_CONTEXT"
        assert ExpectedConclusion.INTENTIONAL.value == "INTENTIONAL"
    
    def test_from_string(self):
        assert ExpectedConclusion("BUG") == ExpectedConclusion.BUG
        assert ExpectedConclusion("CORRECT") == ExpectedConclusion.CORRECT


class TestDifficulty:
    def test_enum_values(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"
    
    def test_from_string(self):
        assert Difficulty("easy") == Difficulty.EASY
        assert Difficulty("medium") == Difficulty.MEDIUM
        assert Difficulty("hard") == Difficulty.HARD


class TestCasePlan:
    @pytest.fixture
    def sample_plan(self):
        return CasePlan(
            title="Test Plan",
            language="Python",
            technical_domain="error handling",
            primary_concept="test concept",
            failure_mechanism="exception not caught",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="test_sig",
            case_summary="tests exception handling",
        )
    
    def test_to_dict(self, sample_plan):
        data = sample_plan.to_dict()
        
        assert data["title"] == "Test Plan"
        assert data["language"] == "Python"
        assert data["technical_domain"] == "error handling"
        assert data["primary_concept"] == "test concept"
        assert data["failure_mechanism"] == "exception not caught"
        assert data["expected_conclusion"] == "BUG"
        assert data["difficulty"] == "medium"
        assert data["code_shape"] == "function"
        assert data["semantic_signature"] == "test_sig"
        assert data["case_summary"] == "tests exception handling"
    
    def test_from_dict(self):
        data = {
            "title": "Dict Plan",
            "language": "Java",
            "technical_domain": "validation",
            "primary_concept": "dict concept",
            "failure_mechanism": "null pointer",
            "expected_conclusion": "CORRECT",
            "difficulty": "easy",
            "code_shape": "class",
            "semantic_signature": "dict_sig",
            "case_summary": "tests null handling",
        }
        
        plan = CasePlan.from_dict(data)
        
        assert plan.title == "Dict Plan"
        assert plan.language == "Java"
        assert plan.technical_domain == "validation"
        assert plan.primary_concept == "dict concept"
        assert plan.failure_mechanism == "null pointer"
        assert plan.expected_conclusion == ExpectedConclusion.CORRECT
        assert plan.difficulty == Difficulty.EASY
        assert plan.code_shape == "class"
        assert plan.semantic_signature == "dict_sig"
        assert plan.case_summary == "tests null handling"
    
    def test_round_trip(self, sample_plan):
        data = sample_plan.to_dict()
        restored = CasePlan.from_dict(data)
        
        assert restored.title == sample_plan.title
        assert restored.language == sample_plan.language
        assert restored.expected_conclusion == sample_plan.expected_conclusion
        assert restored.difficulty == sample_plan.difficulty


class TestGeneratedCase:
    @pytest.fixture
    def sample_plan(self):
        return CasePlan(
            title="Sample Plan",
            language="Go",
            technical_domain="concurrency",
            primary_concept="sample concept",
            failure_mechanism="race condition",
            expected_conclusion=ExpectedConclusion.UNSUPPORTED,
            difficulty=Difficulty.HARD,
            code_shape="module",
            semantic_signature="sample_sig",
            case_summary="tests concurrency",
        )
    
    @pytest.fixture
    def sample_case(self, sample_plan):
        return GeneratedCase(
            filename="001_sample.md",
            markdown_content="# Sample\n```go\ncode\n```",
            markdown_sha256="abc123def456",
            plan=sample_plan,
            ground_truth_explanation="This is the ground truth.",
            evidence_description="Evidence description here.",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="qwen-7b",
        )
    
    def test_to_dict(self, sample_case):
        data = sample_case.to_dict()
        
        assert data["filename"] == "001_sample.md"
        assert data["markdown_sha256"] == "abc123def456"
        assert data["title"] == "Sample Plan"
        assert data["language"] == "Go"
        assert data["technical_domain"] == "concurrency"
        assert data["primary_concept"] == "sample concept"
        assert data["failure_mechanism"] == "race condition"
        assert data["expected_conclusion"] == "UNSUPPORTED"
        assert data["difficulty"] == "hard"
        assert data["semantic_signature"] == "sample_sig"
        assert data["ground_truth_explanation"] == "This is the ground truth."
        assert data["evidence_description"] == "Evidence description here."
        assert data["timestamp"] == "2024-01-01T00:00:00Z"
        assert data["model_identifier"] == "qwen-7b"
    
    def test_from_dict(self):
        data = {
            "filename": "002_restored.md",
            "markdown_sha256": "xyz789",
            "title": "Restored Plan",
            "language": "Rust",
            "technical_domain": "ownership",
            "primary_concept": "restored concept",
            "failure_mechanism": "borrow error",
            "expected_conclusion": "NEEDS_CONTEXT",
            "difficulty": "hard",
            "semantic_signature": "restored_sig",
            "ground_truth_explanation": "Restored explanation.",
            "evidence_description": "Restored evidence.",
            "timestamp": "2024-06-15T12:00:00Z",
            "model_identifier": "qwen-14b",
        }
        
        case = GeneratedCase.from_dict(data)
        
        assert case.filename == "002_restored.md"
        assert case.markdown_sha256 == "xyz789"
        assert case.plan.title == "Restored Plan"
        assert case.plan.language == "Rust"
        assert case.plan.technical_domain == "ownership"
        assert case.plan.primary_concept == "restored concept"
        assert case.plan.expected_conclusion == ExpectedConclusion.NEEDS_CONTEXT
        assert case.ground_truth_explanation == "Restored explanation."
    
    def test_from_dict_with_optional_fields(self):
        data = {
            "filename": "003_partial.md",
            "markdown_sha256": "partial123",
            "title": "Partial Plan",
            "language": "C++",
            "technical_domain": "memory",
            "primary_concept": "partial concept",
            "failure_mechanism": "use after free",
            "expected_conclusion": "INTENTIONAL",
            "difficulty": "medium",
            "semantic_signature": "partial_sig",
            "ground_truth_explanation": "Partial explanation.",
            "evidence_description": "Partial evidence.",
            "timestamp": "2024-01-01T00:00:00Z",
            "model_identifier": "qwen-32b",
            # code_shape is optional in from_dict
        }
        
        case = GeneratedCase.from_dict(data)
        assert case.plan.code_shape == ""  # Default value
    
    def test_to_dict_includes_markdown_content(self, sample_case):
        """Persisted cases retain Markdown for preview and diversity checks."""
        data = sample_case.to_dict()

        assert data["markdown_content"] == sample_case.markdown_content
        assert "markdown_sha256" in data

    def test_from_dict_without_markdown_marks_legacy_content_unavailable(self):
        data = {
            "filename": "004_legacy.md",
            "markdown_sha256": "legacy123",
            "title": "Legacy",
            "language": "Python",
            "technical_domain": "validation",
            "primary_concept": "legacy concept",
            "failure_mechanism": "unknown",
            "expected_conclusion": "BUG",
            "difficulty": "easy",
            "semantic_signature": "legacy_sig",
            "ground_truth_explanation": "Legacy explanation.",
            "evidence_description": "Legacy evidence.",
            "timestamp": "2024-01-01T00:00:00Z",
            "model_identifier": "legacy-model",
        }

        restored = GeneratedCase.from_dict(data)

        assert restored.markdown_content is None
    
    def test_ground_truth_not_in_markdown(self, sample_case):
        """Verify ground truth is stored separately from markdown."""
        # The markdown content should not contain ground truth info
        assert "ground_truth" not in sample_case.markdown_content.lower()
        assert sample_case.ground_truth_explanation != ""
