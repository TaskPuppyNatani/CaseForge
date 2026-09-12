"""Tests for diversity validation and similarity checking."""

import pytest
from benchmark_case_generator.diversity import (
    normalize_text,
    extract_tokens,
    jaccard_similarity,
    sequence_similarity,
    compute_code_similarity,
    concepts_collide,
    DiversityTracker,
    extract_code_from_markdown,
    INITIAL_CONCEPTS,
)
from benchmark_case_generator.models import CasePlan, GeneratedCase, ExpectedConclusion, Difficulty
from benchmark_case_generator.storage import GeneratorState


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"
    
    def test_whitespace(self):
        assert normalize_text("  multiple   spaces  ") == "multiple spaces"
    
    def test_combined(self):
        assert normalize_text("  HELLO   World  ") == "hello world"


class TestExtractTokens:
    def test_basic(self):
        tokens = extract_tokens("hello world foo bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
    
    def test_filters_short(self):
        tokens = extract_tokens("a an the hello")
        assert "a" not in tokens
        assert "an" not in tokens
        assert "the" in tokens  # 3 chars
        assert "hello" in tokens
    
    def test_removes_punctuation(self):
        tokens = extract_tokens("hello; world, foo{}")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens


class TestJaccardSimilarity:
    def test_identical(self):
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0
    
    def test_disjoint(self):
        s1 = {"a", "b"}
        s2 = {"c", "d"}
        assert jaccard_similarity(s1, s2) == 0.0
    
    def test_partial(self):
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        # intersection: {b, c} = 2, union: {a, b, c, d} = 4
        assert abs(jaccard_similarity(s1, s2) - 0.5) < 0.01
    
    def test_empty(self):
        assert jaccard_similarity(set(), set()) == 1.0
        assert jaccard_similarity({"a"}, set()) == 0.0


class TestSequenceSimilarity:
    def test_identical(self):
        assert sequence_similarity("hello", "hello") == 1.0
    
    def test_different(self):
        assert sequence_similarity("abc", "xyz") < 0.5
    
    def test_partial(self):
        # "hello" vs "hallo" should be high similarity
        assert sequence_similarity("hello", "hallo") > 0.7


class TestComputeCodeSimilarity:
    def test_identical_code(self):
        code = "def foo(): return 1"
        assert compute_code_similarity(code, code) > 0.9
    
    def test_different_code(self):
        code1 = "def foo(): return 1"
        code2 = "class Bar { public void baz() {} }"
        assert compute_code_similarity(code1, code2) < 0.5
    
    def test_variable_rename(self):
        code1 = "def process(data): return data + 1"
        code2 = "def process(items): return items + 1"
        # Should still be similar but not identical
        sim = compute_code_similarity(code1, code2)
        assert 0.5 < sim < 1.0


class TestMarkdownCodeExtraction:
    @pytest.mark.parametrize("tag", ["python", "c++", "c#", "javascript", "rust", "tsx:strict"])
    def test_accepts_punctuation_bearing_info_strings(self, tag):
        assert extract_code_from_markdown(f"```{tag}\nvalue()\n```") == "value()\n"

    def test_accepts_untagged_fence(self):
        assert extract_code_from_markdown("```\nvalue()\n```") == "value()\n"

    @pytest.mark.parametrize(
        "markdown",
        [
            "```python\n\n```",
            "```python\nvalue()",
            "inline `value()` only",
        ],
    )
    def test_requires_non_empty_closed_fence(self, markdown):
        assert extract_code_from_markdown(markdown) is None


class TestConceptsCollide:
    def test_exact_concept_match(self):
        collide, reason = concepts_collide(
            "null check after access",
            "NPE risk",
            "check_after_use",
            "null check after access",
            "NPE risk",
            "check_after_use",
        )
        assert collide is True
        assert "duplicates" in reason.lower() or "similar" in reason.lower()
    
    def test_different_concepts(self):
        collide, reason = concepts_collide(
            "resource not closed",
            "resource leak",
            "unclosed_resource",
            "null check missing",
            "NPE risk",
            "missing_null_check",
        )
        assert collide is False
        assert reason == ""
    
    def test_similar_concepts(self):
        collide, reason = concepts_collide(
            "buffer overflow in string copy",
            "memory corruption",
            "buffer_overflow_string",
            "buffer overflow in string copy",  # Nearly identical
            "memory corruption",
            "buffer_overflow_string",
        )
        assert collide is True


class TestDiversityTracker:
    def test_initial_concepts_loaded(self):
        tracker = DiversityTracker()
        # Initial concepts should be in the set
        assert len(tracker._concepts) >= len(INITIAL_CONCEPTS)
    
    def test_load_cases(self):
        tracker = DiversityTracker()
        
        case = GeneratedCase(
            filename="001_test.md",
            markdown_content="# Test\n```python\ncode\n```",
            markdown_sha256="abc123",
            plan=CasePlan(
                title="Test Case",
                language="Python",
                technical_domain="error handling",
                primary_concept="unique concept xyz",
                failure_mechanism="test mechanism",
                expected_conclusion=ExpectedConclusion.BUG,
                difficulty=Difficulty.MEDIUM,
                code_shape="function",
                semantic_signature="unique_sig",
                case_summary="summary",
            ),
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        
        tracker.load_cases([case])
        assert tracker.case_count == 1
        
        # Concept should be tracked
        is_valid, reason = tracker.validate_plan(case.plan)
        assert is_valid is False  # Should reject duplicate
    
    def test_validate_new_plan(self):
        tracker = DiversityTracker()
        
        plan = CasePlan(
            title="New Case",
            language="Java",
            technical_domain="concurrency",
            primary_concept="completely new concept abc123",
            failure_mechanism="race condition",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.HARD,
            code_shape="class",
            semantic_signature="new_sig_xyz",
            case_summary="summary",
        )
        
        is_valid, reason = tracker.validate_plan(plan)
        assert is_valid is True
        assert reason == ""
    
    def test_add_case_updates_tracker(self):
        tracker = DiversityTracker()
        
        plan = CasePlan(
            title="Added Case",
            language="Go",
            technical_domain="resource management",
            primary_concept="added concept def456",
            failure_mechanism="leak",
            expected_conclusion=ExpectedConclusion.CORRECT,
            difficulty=Difficulty.EASY,
            code_shape="module",
            semantic_signature="added_sig",
            case_summary="summary",
        )
        
        case = GeneratedCase(
            filename="001_added.md",
            markdown_content="# Added\n```go\ncode\n```",
            markdown_sha256="def456",
            plan=plan,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        
        assert tracker.case_count == 0
        tracker.add_case(case)
        assert tracker.case_count == 1
    
    def test_get_existing_concepts_summary(self):
        tracker = DiversityTracker()
        
        # Empty tracker
        summary = tracker.get_existing_concepts_summary()
        assert "No cases generated yet" in summary
        
        # With cases
        plan = CasePlan(
            title="Summary Test",
            language="Python",
            technical_domain="validation",
            primary_concept="summary concept",
            failure_mechanism="validation failure",
            expected_conclusion=ExpectedConclusion.UNSUPPORTED,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="sig",
            case_summary="summary",
        )
        
        case = GeneratedCase(
            filename="001_summary.md",
            markdown_content="# Summary\n```python\ncode\n```",
            markdown_sha256="sum123",
            plan=plan,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        
        tracker.add_case(case)
        summary = tracker.get_existing_concepts_summary()
        assert "summary concept" in summary

    def test_reloaded_markdown_reconstructs_code_diversity(self, tmp_path):
        plan = CasePlan(
            title="Persisted Code",
            language="Python",
            technical_domain="validation",
            primary_concept="persisted code concept",
            failure_mechanism="validation failure",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="persisted-code-signature",
            case_summary="summary",
        )
        markdown = "# Persisted\n```python\ndef check(value):\n    return value + 1\n```"
        case = GeneratedCase(
            filename="001_persisted.md",
            markdown_content=markdown,
            markdown_sha256="persisted-sha",
            plan=plan,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="test-model",
        )
        state_file = tmp_path / "state.json"
        first = GeneratorState(str(state_file))
        first.initialize("test-model")
        first.add_case(case)
        first.save()

        restored = GeneratorState(str(state_file)).load()
        assert restored[0].markdown_content == markdown

        tracker = DiversityTracker()
        tracker.load_cases(restored)
        candidate = CasePlan(
            title="Different Concept",
            language="Python",
            technical_domain="concurrency",
            primary_concept="different concept after reload",
            failure_mechanism="different mechanism",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="different-signature",
            case_summary="summary",
        )
        valid, reason = tracker.validate_plan(candidate, "def check(value):\n    return value + 1\n")

        assert valid is False
        assert "code sample" in reason.lower()
