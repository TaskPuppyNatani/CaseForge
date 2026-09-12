"""Tests for the generation module with mocked API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import math
import tempfile
from pathlib import Path

from benchmark_case_generator.generation import (
    CaseGenerator,
    DEFAULT_CONCLUSION_DISTRIBUTION,
    _select_conclusion,
    validate_conclusion_distribution,
)
from benchmark_case_generator.models import (
    CasePlan, GeneratedCase, ExpectedConclusion, Difficulty
)
from benchmark_case_generator.client import ClientConfig, ModelClient
from benchmark_case_generator.diversity import DiversityTracker
from benchmark_case_generator.storage import GeneratorState, OutputManager


class MockModelClient:
    """Mock client that returns predetermined responses."""
    
    def __init__(self, plan_responses=None, case_responses=None):
        self.plan_responses = plan_responses or []
        self.case_responses = case_responses or []
        self.plan_call_count = 0
        self.case_call_count = 0
        self.requests = []
        self.config = ClientConfig(model="mock-model")
    
    def chat_completion(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        # Check if this is a plan request (mentions "case plan as JSON")
        user_content = messages[-1]["content"] if messages else ""
        
        if (
            "case plan as JSON" in user_content
            or "Generate ONE benchmark" in user_content
            or "previous case plan was rejected" in user_content
        ):
            response_text = self.plan_responses[self.plan_call_count % len(self.plan_responses)]
            self.plan_call_count += 1
        else:
            response_text = self.case_responses[self.case_call_count % len(self.case_responses)]
            self.case_call_count += 1
        
        return {"choices": [{"message": {"content": response_text}}]}
    
    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


def create_test_plan():
    return CasePlan(
        title="Test Plan",
        language="Python",
        technical_domain="error handling",
        primary_concept="test concept abc123",
        failure_mechanism="test mechanism",
        expected_conclusion=ExpectedConclusion.BUG,
        difficulty=Difficulty.MEDIUM,
        code_shape="function",
        semantic_signature="test_sig_xyz",
        case_summary="summary",
    )


def create_test_markdown(plan):
    return f"""Review the supplied code for concrete defects.

Report only findings directly supported by the visible code.

Code sample:

```python
def example():
    # {plan.primary_concept}
    return None
```
"""


def make_plan_json(
    *,
    title: str,
    concept: str,
    language: str = "Python",
    conclusion: str = "BUG",
    difficulty: str = "medium",
) -> str:
    return json.dumps({
        "title": title,
        "language": language,
        "technical_domain": "validation",
        "primary_concept": concept,
        "failure_mechanism": "test mechanism",
        "expected_conclusion": conclusion,
        "difficulty": difficulty,
        "code_shape": "function",
        "semantic_signature": f"{concept}-signature",
        "case_summary": "tests target handling",
    })


class TestGenerationWithMock:
    @pytest.fixture
    def setup_components(self, tmp_path):
        """Set up generator components."""
        output_dir = tmp_path / "output"
        state_file = tmp_path / "state.json"
        
        config = ClientConfig()
        client = ModelClient(config)
        
        state = GeneratorState(str(state_file))
        state.initialize("mock-model")
        
        output = OutputManager(str(output_dir))
        output.initialize()
        
        tracker = DiversityTracker()
        
        return client, tracker, output, state
    
    def test_generate_single_case_success(self, setup_components):
        client, tracker, output, state = setup_components
        
        plan_json = json.dumps({
            "title": "Unique Test Case",
            "language": "Python",
            "technical_domain": "error handling",
            "primary_concept": "unique concept xyz789",
            "failure_mechanism": "exception not handled",
            "expected_conclusion": "BUG",
            "difficulty": "medium",
            "code_shape": "function",
            "semantic_signature": "unique_sig_abc",
            "case_summary": "tests exception handling",
        })
        
        markdown = create_test_markdown(create_test_plan())
        
        mock_client = MockModelClient(
            plan_responses=[plan_json],
            case_responses=[markdown]
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        assert len(cases) == 1
        assert cases[0].plan.primary_concept == "unique concept xyz789"
    
    def test_retry_on_parse_failure(self, setup_components):
        client, tracker, output, state = setup_components
        
        # First response is invalid JSON, second is valid
        invalid_response = "This is not JSON"
        valid_plan = json.dumps({
            "title": "Retry Test",
            "language": "Java",
            "technical_domain": "validation",
            "primary_concept": "retry concept def456",
            "failure_mechanism": "null check missing",
            "expected_conclusion": "CORRECT",
            "difficulty": "easy",
            "code_shape": "class",
            "semantic_signature": "retry_sig",
            "case_summary": "tests null handling",
        })
        
        markdown = create_test_markdown(create_test_plan())
        
        mock_client = MockModelClient(
            plan_responses=[invalid_response, valid_plan],
            case_responses=[markdown]
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.CORRECT: 1.0},
            languages=["Java"],
            difficulty=Difficulty.EASY,
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        assert len(cases) == 1
        assert mock_client.plan_call_count >= 2  # Should have retried
    
    def test_retry_on_diversity_collision(self, setup_components):
        client, tracker, output, state = setup_components
        
        # Pre-populate tracker with a concept
        existing_plan = CasePlan(
            title="Existing",
            language="Python",
            technical_domain="error handling",
            primary_concept="collision concept",
            failure_mechanism="test",
            expected_conclusion=ExpectedConclusion.BUG,
            difficulty=Difficulty.MEDIUM,
            code_shape="function",
            semantic_signature="existing_sig",
            case_summary="summary",
        )
        
        existing_case = GeneratedCase(
            filename="001_existing.md",
            markdown_content="# Existing",
            markdown_sha256="abc",
            plan=existing_plan,
            ground_truth_explanation="explanation",
            evidence_description="evidence",
            timestamp="2024-01-01T00:00:00Z",
            model_identifier="mock",
        )
        tracker.add_case(existing_case)
        
        # First plan collides, second is unique
        collision_plan = json.dumps({
            "title": "Collision",
            "language": "Python",
            "technical_domain": "error handling",
            "primary_concept": "collision concept",  # Same as existing
            "failure_mechanism": "test",
            "expected_conclusion": "BUG",
            "difficulty": "medium",
            "code_shape": "function",
            "semantic_signature": "collision_sig",
            "case_summary": "summary",
        })
        
        unique_plan = json.dumps({
            "title": "Unique After Collision",
            "language": "Python",
            "technical_domain": "concurrency",
            "primary_concept": "unique after collision ghi789",
            "failure_mechanism": "race condition",
            "expected_conclusion": "BUG",
            "difficulty": "medium",
            "code_shape": "module",
            "semantic_signature": "unique_after_collision",
            "case_summary": "tests concurrency",
        })
        
        markdown1 = "# Collision\n```go\ncode\n```"
        markdown2 = "# Unique\n```python\ncode\n```"
        
        mock_client = MockModelClient(
            plan_responses=[collision_plan, unique_plan],
            case_responses=[markdown1, markdown2]
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        assert len(cases) == 1
        assert "unique after collision" in cases[0].plan.primary_concept.lower()
        assert mock_client.plan_call_count == 2
        full_case_prompts = [
            request[-1]["content"]
            for request, _kwargs in mock_client.requests
            if request and "Create a complete benchmark test case" in request[-1]["content"]
        ]
        assert len(full_case_prompts) == 1
        assert "unique after collision ghi789" in full_case_prompts[0]
    
    def test_respects_max_retries(self, setup_components):
        client, tracker, output, state = setup_components
        
        # Always return invalid JSON
        invalid_response = "Not JSON at all"
        
        mock_client = MockModelClient(
            plan_responses=[invalid_response],
            case_responses=[]
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=2,
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        # Should give up after retries
        assert len(cases) == 0
        assert mock_client.plan_call_count == 2

    def test_diversity_recovery_remains_bounded_when_revisions_fail(self, setup_components):
        client, tracker, output, state = setup_components
        colliding_plan = json.dumps({
            "title": "Still Colliding",
            "language": "Python",
            "technical_domain": "resource management",
            "primary_concept": "resource/stream is not closed",
            "failure_mechanism": "resource leak",
            "expected_conclusion": "BUG",
            "difficulty": "medium",
            "code_shape": "function",
            "semantic_signature": "same_signature",
            "case_summary": "still collides",
        })
        mock_client = MockModelClient(
            plan_responses=[colliding_plan],
            case_responses=[],
        )
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=4,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        assert generator.generate_cases(1, dry_run=True) == []
        assert mock_client.plan_call_count == 4
        assert mock_client.case_call_count == 0

    def test_generation_accepts_safe_answer_label_substrings(self, setup_components):
        client, tracker, output, state = setup_components
        plan_json = make_plan_json(
            title="Safe Label Substrings",
            concept="safe answer label substring concept",
        )
        markdown = '''# Review
```python
print("debug:")
status = "nondefect:"
```
'''
        mock_client = MockModelClient(
            plan_responses=[plan_json],
            case_responses=[markdown],
        )
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=2,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        cases = generator.generate_cases(1, dry_run=True)

        assert len(cases) == 1
        assert mock_client.plan_call_count == 1
        assert mock_client.case_call_count == 1
    
    def test_dry_run_does_not_write_files(self, setup_components, tmp_path):
        client, tracker, output, state = setup_components
        
        plan_json = json.dumps({
            "title": "Dry Run Test",
            "language": "Rust",
            "technical_domain": "ownership",
            "primary_concept": "dry run concept jkl012",
            "failure_mechanism": "borrow checker error",
            "expected_conclusion": "NEEDS_CONTEXT",
            "difficulty": "hard",
            "code_shape": "struct",
            "semantic_signature": "dry_run_sig",
            "case_summary": "tests ownership",
        })
        
        markdown = "# Dry Run\n```rust\ncode\n```"
        
        mock_client = MockModelClient(
            plan_responses=[plan_json],
            case_responses=[markdown]
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.NEEDS_CONTEXT: 1.0},
            languages=["Rust"],
            difficulty=Difficulty.HARD,
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        assert len(cases) == 1
        # In dry run, markdown_content should be set but no file written
        # The sha256 should be empty
        assert cases[0].markdown_sha256 == ""
    
    def test_generates_multiple_cases(self, setup_components):
        client, tracker, output, state = setup_components
        
        plans = []
        markdowns = []
        
        # Use genuinely different technical domains and concepts to pass diversity check
        # Also provide DISTINCT code samples to avoid code similarity rejection
        test_configs = [
            (
                "error handling", 
                "exception swallowed silently in logger", 
                "exc_swallow_001",
                "# Test exception handling\ndef log_error(msg):\n    try:\n        process(msg)\n    except Exception:\n        pass  # Swallowed!\n"
            ),
            (
                "concurrency", 
                "race condition in shared counter increment without lock", 
                "race_counter_002",
                "# Test race condition\ncounter = 0\ndef increment():\n    global counter\n    counter += 1  # Not atomic!\n"
            ),
            (
                "resource management", 
                "file handle leaked when exception thrown before close", 
                "file_leak_003",
                "# Test resource leak\ndef read_file(path):\n    f = open(path)\n    data = f.read()  # May raise!\n    f.close()  # Never reached on exception\n    return data\n"
            ),
        ]
        
        for domain, concept, sig, code in test_configs:
            plans.append(json.dumps({
                "title": f"Test {concept}",
                "language": "Python",
                "technical_domain": domain,
                "primary_concept": concept,
                "failure_mechanism": "test mechanism",
                "expected_conclusion": "BUG",
                "difficulty": "medium",
                "code_shape": "function",
                "semantic_signature": sig,
                "case_summary": f"tests {domain}",
            }))
            markdowns.append(f"# Test {concept}\n```python\n{code}```")
        
        mock_client = MockModelClient(
            plan_responses=plans,
            case_responses=markdowns
        )
        
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )
        
        cases = generator.generate_cases(3, dry_run=True)
        
        assert len(cases) == 3, f"Expected 3 cases, got {len(cases)}"
        concepts = [c.plan.primary_concept for c in cases]
        # All concepts should be unique
        assert len(set(concepts)) == 3, f"Expected 3 unique concepts, got: {concepts}"


class TestPlanTargetValidation:
    @pytest.fixture
    def setup_components(self, tmp_path):
        """Set up generator components for plan-target regressions."""
        output_dir = tmp_path / "output"
        state_file = tmp_path / "state.json"

        client = ModelClient(ClientConfig())
        state = GeneratorState(str(state_file))
        state.initialize("mock-model")
        output = OutputManager(str(output_dir))
        output.initialize()
        return client, DiversityTracker(), output, state

    @pytest.mark.parametrize(
        "wrong_fields",
        [
            {"conclusion": "CORRECT"},
            {"language": "Java"},
            {"difficulty": "hard"},
            {"conclusion": "CORRECT", "language": "Java", "difficulty": "hard"},
        ],
    )
    def test_mismatched_target_is_rejected_and_corrected(
        self,
        setup_components,
        wrong_fields,
    ):
        client, tracker, output, state = setup_components
        wrong = make_plan_json(
            title="Wrong Target",
            concept="wrong target concept",
            **wrong_fields,
        )
        corrected = make_plan_json(
            title="Correct Target",
            concept="corrected target concept",
        )
        mock_client = MockModelClient(
            plan_responses=[wrong, corrected],
            case_responses=["# Case\n```python\nreturn 1\n```"],
        )
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        cases = generator.generate_cases(1, dry_run=True)

        assert len(cases) == 1
        assert cases[0].plan.primary_concept == "corrected target concept"
        assert mock_client.plan_call_count == 2
        recovery_prompt = mock_client.requests[1][0][-1]["content"]
        assert "requested" in recovery_prompt

    def test_target_mismatch_exhaustion_is_bounded(self, setup_components):
        client, tracker, output, state = setup_components
        wrong = make_plan_json(
            title="Always Wrong",
            concept="always wrong target",
            language="Java",
            conclusion="CORRECT",
            difficulty="hard",
        )
        mock_client = MockModelClient(
            plan_responses=[wrong],
            case_responses=[],
        )
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        assert generator.generate_cases(1, dry_run=True) == []
        assert mock_client.plan_call_count == 3
        assert mock_client.case_call_count == 0


    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("language", None),
            ("language", 123),
            ("difficulty", None),
            ("difficulty", {"level": "medium"}),
            ("conclusion", {"value": "BUG"}),
        ],
    )
    def test_malformed_plan_fields_recover_inside_bounded_generation(
        self,
        setup_components,
        field,
        value,
    ):
        client, tracker, output, state = setup_components
        malformed_fields = {
            "language": "Python",
            "conclusion": "BUG",
            "difficulty": "medium",
        }
        malformed_fields[field] = value
        malformed = make_plan_json(
            title="Malformed Plan",
            concept="malformed plan concept",
            **malformed_fields,
        )
        corrected = make_plan_json(
            title="Corrected Plan",
            concept="corrected malformed plan concept",
        )
        mock_client = MockModelClient(
            plan_responses=[malformed, corrected],
            case_responses=["# Case\n```python\nreturn 1\n```"],
        )
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=2,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        cases = generator.generate_cases(1, dry_run=True)

        assert len(cases) == 1
        assert cases[0].plan.primary_concept == "corrected malformed plan concept"
        assert cases[0].plan.language == "Python"
        assert cases[0].plan.difficulty is Difficulty.MEDIUM
        assert cases[0].plan.expected_conclusion is ExpectedConclusion.BUG
        assert mock_client.plan_call_count == 2
        assert mock_client.case_call_count == 1

    def test_persistent_malformed_plan_exhaustion_is_bounded(self, setup_components):
        client, tracker, output, state = setup_components
        malformed = make_plan_json(
            title="Malformed Plan",
            concept="persistent malformed plan",
            language=None,
        )
        mock_client = MockModelClient(plan_responses=[malformed], case_responses=[])
        generator = CaseGenerator(
            client=mock_client,
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=3,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        assert generator.generate_cases(1, dry_run=True) == []
        assert mock_client.plan_call_count == 3
        assert mock_client.case_call_count == 0

    def test_provider_plan_error_remains_distinct_from_malformed_content(
        self,
        setup_components,
        capsys,
    ):
        _client, tracker, output, state = setup_components

        class FailingPlanClient:
            config = ClientConfig(model="mock-model")

            def chat_completion(self, messages, **kwargs):
                raise RuntimeError("transport unavailable")

        generator = CaseGenerator(
            client=FailingPlanClient(),
            diversity_tracker=tracker,
            output_manager=output,
            state=state,
            max_retries=2,
            conclusion_distribution={ExpectedConclusion.BUG: 1.0},
            languages=["Python"],
            difficulty=Difficulty.MEDIUM,
        )

        assert generator.generate_cases(1, dry_run=True) == []
        captured = capsys.readouterr().out
        assert "Plan generation error" in captured
        assert "Plan validation error" not in captured


class TestConclusionDistribution:
    def test_rejects_all_zero_weights(self):
        distribution = {conclusion: 0 for conclusion in ExpectedConclusion}

        with pytest.raises(ValueError, match="positive total"):
            validate_conclusion_distribution(distribution)
        with pytest.raises(ValueError, match="positive total"):
            _select_conclusion(distribution)

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_conclusion_distribution({ExpectedConclusion.BUG: -1})

    @pytest.mark.parametrize("weight", ["not numeric", math.nan, math.inf, -math.inf])
    def test_rejects_nonnumeric_and_nonfinite_weights(self, weight):
        with pytest.raises(ValueError, match="finite|numeric"):
            validate_conclusion_distribution({ExpectedConclusion.BUG: weight})

    def test_rejects_integer_float_conversion_overflow(self):
        with pytest.raises(ValueError, match="finite|numeric"):
            validate_conclusion_distribution({ExpectedConclusion.BUG: 10**400})

    def test_rejects_nonfinite_aggregate_from_finite_weights(self):
        distribution = {
            ExpectedConclusion.BUG: 1e308,
            ExpectedConclusion.CORRECT: 1e308,
        }

        with pytest.raises(ValueError, match="finite"):
            validate_conclusion_distribution(distribution)
        with pytest.raises(ValueError, match="finite"):
            _select_conclusion(distribution)

    def test_accepts_exact_hundred_and_relative_non_hundred_weights(self):
        exact = validate_conclusion_distribution({
            ExpectedConclusion.BUG: 40,
            ExpectedConclusion.CORRECT: 30,
            ExpectedConclusion.UNSUPPORTED: 15,
            ExpectedConclusion.NEEDS_CONTEXT: 10,
            ExpectedConclusion.INTENTIONAL: 5,
        })
        relative = validate_conclusion_distribution({
            ExpectedConclusion.BUG: 1,
            ExpectedConclusion.INTENTIONAL: 3,
        })

        assert exact[ExpectedConclusion.BUG] == 40.0
        assert relative[ExpectedConclusion.INTENTIONAL] == 3.0

    def test_single_nonzero_weight_and_boundaries_do_not_fall_through(self):
        with patch("random.random", return_value=0.999999):
            assert _select_conclusion({ExpectedConclusion.BUG: 1}) is ExpectedConclusion.BUG
            assert _select_conclusion({
                ExpectedConclusion.BUG: 1,
                ExpectedConclusion.INTENTIONAL: 0,
            }) is ExpectedConclusion.BUG

        with patch("random.random", return_value=0.0):
            assert _select_conclusion({
                ExpectedConclusion.BUG: 1,
                ExpectedConclusion.INTENTIONAL: 3,
            }) is ExpectedConclusion.BUG

        with patch("random.random", return_value=0.75):
            assert _select_conclusion({
                ExpectedConclusion.BUG: 1,
                ExpectedConclusion.INTENTIONAL: 3,
            }) is ExpectedConclusion.INTENTIONAL


class TestMarkdownValidation:
    @pytest.fixture
    def generator(self):
        # The validator is pure; construct the production class without
        # substituting its implementation with a test double.
        return CaseGenerator.__new__(CaseGenerator)

    def test_validates_code_block_present(self, generator):
        assert generator._validate_markdown("# Test\n```python\ncode\n```") is True
        assert generator._validate_markdown("# Test without code") is False

    @pytest.mark.parametrize(
        "leak",
        [
            "BUG: off-by-one error",
            "Bug: off-by-one error",
            "bug: off-by-one error",
            "DEFECT: missing bounds check",
            "Defect: missing bounds check",
            "defect: missing bounds check",
            "**BUG:** off-by-one error",
            "- BUG: resource leak",
            "(DEFECT: null dereference)",
        ],
    )
    def test_rejects_answer_leaks_case_insensitively(self, generator, leak):
        markdown = f"# Test\n{leak}\n```python\ncode\n```"

        assert generator._validate_markdown(markdown) is False

    @pytest.mark.parametrize(
        "safe_content",
        [
            "debug: true",
            "Debug: enabled",
            'print("debug:")',
            "mybug:",
            "nonbug:",
            "nondefect:",
            "nodefect:",
            "some_defect:",
        ],
    )
    def test_preserves_safe_answer_label_substrings(self, generator, safe_content):
        markdown = f"# Review\n```python\n{safe_content}\n```"

        assert generator._validate_markdown(markdown) is True

    @pytest.mark.parametrize("tag", ["python", "c++", "c#", "javascript", "java", "rust", "tsx:strict"])
    def test_accepts_punctuation_bearing_fence_info_strings(self, generator, tag):
        assert generator._validate_markdown(f"# Test\n```{tag}\ncode\n```") is True

    def test_accepts_untagged_fence(self, generator):
        assert generator._validate_markdown("# Test\n```\ncode\n```") is True

    @pytest.mark.parametrize(
        "markdown",
        [
            "# Test\n```python\n\n```",
            "# Test\n```python\ncode",
            "# Test with `inline` backticks only",
        ],
    )
    def test_rejects_empty_unclosed_or_inline_backticks(self, generator, markdown):
        assert generator._validate_markdown(markdown) is False
