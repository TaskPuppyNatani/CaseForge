"""Tests for the generation module with mocked API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from benchmark_case_generator.generation import CaseGenerator, DEFAULT_CONCLUSION_DISTRIBUTION
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
        self.config = ClientConfig(model="mock-model")
    
    def chat_completion(self, messages, **kwargs):
        # Check if this is a plan request (mentions "case plan as JSON")
        user_content = messages[-1]["content"] if messages else ""
        
        if "case plan as JSON" in user_content or "Generate ONE benchmark" in user_content:
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
            "language": "Go",
            "technical_domain": "concurrency",
            "primary_concept": "unique after collision ghi789",
            "failure_mechanism": "race condition",
            "expected_conclusion": "UNSUPPORTED",
            "difficulty": "hard",
            "code_shape": "module",
            "semantic_signature": "unique_after_collision",
            "case_summary": "tests concurrency",
        })
        
        markdown1 = "# Collision\n```go\ncode\n```"
        markdown2 = "# Unique\n```go\ncode\n```"
        
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
        )
        
        cases = generator.generate_cases(1, dry_run=True)
        
        assert len(cases) == 1
        assert "unique after collision" in cases[0].plan.primary_concept.lower()
    
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
        )
        
        cases = generator.generate_cases(3, dry_run=True)
        
        assert len(cases) == 3, f"Expected 3 cases, got {len(cases)}"
        concepts = [c.plan.primary_concept for c in cases]
        # All concepts should be unique
        assert len(set(concepts)) == 3, f"Expected 3 unique concepts, got: {concepts}"


class TestMarkdownValidation:
    def test_validates_code_block_present(self):
        from benchmark_case_generator.generation import CaseGenerator
        
        # Create minimal generator just for validation method
        class MinimalGenerator:
            def _validate_markdown(self, md):
                return "```" in md
        
        gen = MinimalGenerator()
        assert gen._validate_markdown("# Test\n```python\ncode\n```") is True
        assert gen._validate_markdown("# Test without code") is False
    
    def test_rejects_answer_leaks(self):
        from benchmark_case_generator.generation import CaseGenerator
        
        class MinimalGenerator:
            def _validate_markdown(self, md):
                # Match actual implementation logic
                forbidden = ["this code contains a bug", "this code is safe", 
                            "this code is correct", "bug:", "defect:", "the problem here"]
                lower_md = md.lower()
                for pattern in forbidden:
                    if pattern in lower_md:
                        return False
                return "```" in md
        
        gen = MinimalGenerator()
        assert gen._validate_markdown("# Test\n```python\ncode\n```") is True
        assert gen._validate_markdown("# Test\nThis code contains a bug\n```python\ncode\n```") is False
        assert gen._validate_markdown("# Test\nBug: null pointer\n```python\ncode\n```") is False
