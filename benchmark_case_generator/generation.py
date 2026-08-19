"""Two-stage benchmark case generation."""

import json
import re
from datetime import datetime
from typing import Optional

from .models import CasePlan, GeneratedCase, ExpectedConclusion, Difficulty
from .client import ModelClient
from .diversity import DiversityTracker
from .storage import OutputManager, GeneratorState


# Default distribution for expected conclusions
DEFAULT_CONCLUSION_DISTRIBUTION = {
    ExpectedConclusion.BUG: 0.40,
    ExpectedConclusion.CORRECT: 0.30,
    ExpectedConclusion.UNSUPPORTED: 0.15,
    ExpectedConclusion.NEEDS_CONTEXT: 0.10,
    ExpectedConclusion.INTENTIONAL: 0.05,
}

TECHNICAL_DOMAINS = [
    "error handling",
    "resource management",
    "bounds/indexing",
    "nullability",
    "immutability",
    "state management",
    "collections",
    "concurrency",
    "locking",
    "async control flow",
    "transactions",
    "SQL/database use",
    "serialization/parsing",
    "caching",
    "filesystem behavior",
    "numeric conversion",
    "ownership/lifetime",
    "API contracts",
    "iterator/stream lifecycle",
    "validation",
    "cleanup",
    "data transformation",
    "exception propagation",
]

LANGUAGES = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", 
    "C#", "C++", "Ruby", "Kotlin", "Swift"
]


def _select_conclusion(distribution: dict[ExpectedConclusion, float]) -> ExpectedConclusion:
    """Select a conclusion based on distribution."""
    import random
    r = random.random()
    cumulative = 0.0
    for conclusion, prob in distribution.items():
        cumulative += prob
        if r <= cumulative:
            return conclusion
    return list(distribution.keys())[-1]


class CaseGenerator:
    """Generates benchmark cases using two-stage approach."""

    def __init__(
        self,
        client: ModelClient,
        diversity_tracker: DiversityTracker,
        output_manager: OutputManager,
        state: GeneratorState,
        conclusion_distribution: Optional[dict[ExpectedConclusion, float]] = None,
        max_retries: int = 5,
        languages: Optional[list[str]] = None,
        difficulty: Optional[Difficulty] = None,
    ):
        self.client = client
        self.tracker = diversity_tracker
        self.output = output_manager
        self.state = state
        self.distribution = conclusion_distribution or DEFAULT_CONCLUSION_DISTRIBUTION
        self.max_retries = max_retries
        self.languages = languages or LANGUAGES
        self.difficulty = difficulty

    def generate_cases(self, count: int, dry_run: bool = False) -> list[GeneratedCase]:
        """Generate the requested number of cases."""
        generated = []
        
        for i in range(count):
            case_num = self.tracker.case_count + i + 1
            print(f"\nGenerating case {case_num}...")
            
            case = self._generate_single_case(dry_run)
            if case:
                generated.append(case)
                self.tracker.add_case(case)
                if not dry_run:
                    self.state.add_case(case)
                    self.state.save()
            else:
                print(f"  Failed to generate case {case_num} after {self.max_retries} retries")
        
        return generated

    def _generate_single_case(self, dry_run: bool = False) -> Optional[GeneratedCase]:
        """Generate a single case through two stages."""
        
        # Select parameters for this case
        conclusion = _select_conclusion(self.distribution)
        language = __import__("random").choice(self.languages)
        difficulty = self.difficulty or __import__("random").choice([Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
        
        plan = None
        
        # STAGE 1: Generate case plan
        for attempt in range(self.max_retries):
            try:
                plan_response = self._request_case_plan(conclusion, language, difficulty)
                plan = self._parse_case_plan(plan_response)
                
                if not plan:
                    print(f"  Plan parse failed (attempt {attempt + 1})")
                    continue
                
                # Validate diversity
                is_valid, reason = self.tracker.validate_plan(plan)
                if not is_valid:
                    print(f"  Diversity check failed: {reason}")
                    # Request a different concept
                    self._request_different_plan(reason, conclusion, language, difficulty)
                    continue
                
                print(f"  Plan accepted: {plan.primary_concept}")
                break
                
            except Exception as e:
                print(f"  Plan generation error (attempt {attempt + 1}): {e}")
                continue
        else:
            return None
        
        # STAGE 2: Generate full benchmark case
        for attempt in range(self.max_retries):
            try:
                markdown = self._request_full_case(plan)
                
                # Validate markdown structure
                if not self._validate_markdown(markdown):
                    print(f"  Markdown validation failed (attempt {attempt + 1})")
                    continue
                
                # Extract code for similarity check
                code_match = re.search(r"```(?:\w+)?\n(.*?)```", markdown, re.DOTALL)
                code_sample = code_match.group(1) if code_match else None
                
                # Final diversity check with code
                is_valid, reason = self.tracker.validate_plan(plan, code_sample)
                if not is_valid:
                    print(f"  Code diversity check failed: {reason}")
                    continue
                
                # Create the case
                now = datetime.utcnow().isoformat() + "Z"
                
                if dry_run:
                    filename = f"{self.tracker.case_count + 1:03d}_{plan.title[:20].lower().replace(' ', '_')}.md"
                    sha256 = ""
                else:
                    filename = self.output.get_next_filename(plan.title)
                    sha256 = self.output.write_case(filename, markdown)
                
                ground_truth = self._extract_ground_truth(plan, markdown)
                evidence = self._describe_evidence(plan, markdown)
                
                case = GeneratedCase(
                    filename=filename,
                    markdown_content=markdown,
                    markdown_sha256=sha256,
                    plan=plan,
                    ground_truth_explanation=ground_truth,
                    evidence_description=evidence,
                    timestamp=now,
                    model_identifier=self.state.model_identifier or self.client.config.model,
                )
                
                print(f"  Generated: {filename}")
                return case
                
            except Exception as e:
                print(f"  Full case error (attempt {attempt + 1}): {e}")
                continue
        
        return None

    def _request_case_plan(
        self,
        conclusion: ExpectedConclusion,
        language: str,
        difficulty: Difficulty,
    ) -> str:
        """Request a case plan from the model."""
        
        existing_summary = self.tracker.get_existing_concepts_summary()
        
        system_prompt = """You are an expert software engineering benchmark designer.
Your task is to create DIVERSITY in benchmark test cases.

Each case must test a DIFFERENT central technical concept.

DO NOT create variations that only change:
- Programming language
- Variable names
- Filenames
- Surface syntax
- Domain nouns

For example, "Python list mutation during iteration" and "Java ArrayList mutation during iteration"
are the SAME concept and should NOT both be included.

Focus on genuinely different failure mechanisms and technical judgments."""

        user_prompt = f"""Generate ONE benchmark case plan as JSON.

Required conclusion type: {conclusion.value}
Target language: {language}
Difficulty: {difficulty.value}

{existing_summary}

Output ONLY valid JSON with these fields:
{{
  "title": "Brief descriptive title",
  "language": "{language}",
  "technical_domain": "One of: {', '.join(TECHNICAL_DOMAINS)}",
  "primary_concept": "The central technical concept being tested",
  "failure_mechanism": "How the defect manifests (or 'N/A' for non-bug cases)",
  "expected_conclusion": "{conclusion.value}",
  "difficulty": "{difficulty.value}",
  "code_shape": "Description of code structure (function, class, module, etc.)",
  "semantic_signature": "Unique semantic pattern that distinguishes this case",
  "case_summary": "2-3 sentence description of what the case tests"
}}

Ensure the primary_concept is SUBSTANTIALLY DIFFERENT from all existing concepts listed above."""

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
        )
        
        return self.client.extract_content(response)

    def _request_different_plan(
        self,
        rejection_reason: str,
        conclusion: ExpectedConclusion,
        language: str,
        difficulty: Difficulty,
    ) -> str:
        """Request a revised plan after rejection."""
        
        system_prompt = """You are revising a benchmark case plan that was rejected for similarity.
Create a COMPLETELY DIFFERENT technical concept."""

        user_prompt = f"""Your previous case plan was rejected:

Rejection reason: {rejection_reason}

Required conclusion type: {conclusion.value}
Target language: {language}
Difficulty: {difficulty.value}

Generate a NEW case plan with a fundamentally different technical concept.
Choose a different technical domain and failure mechanism.

Output ONLY valid JSON with the same structure as before."""

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
        )
        
        return self.client.extract_content(response)

    def _parse_case_plan(self, content: str) -> Optional[CasePlan]:
        """Parse JSON response into CasePlan."""
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        
        try:
            data = json.loads(match.group(0))
            return CasePlan.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return None

    def _request_full_case(self, plan: CasePlan) -> str:
        """Request the full benchmark case markdown."""
        
        system_prompt = """You are creating a self-contained code-review benchmark test case.

FORMAT REQUIREMENTS:
1. Start with the standard review instructions
2. Include any contract/context needed
3. Include the complete code sample in a fenced code block
4. DO NOT include the expected answer or classification
5. DO NOT label the code as "buggy" or "safe"
6. DO NOT give hints about the expected conclusion

The code should be syntactically plausible and realistic.

For BUG cases: There must be a concrete, visible defect.
For CORRECT cases: The suspected defect must actually be handled.
For UNSUPPORTED cases: The issue must depend on unstated assumptions.
For NEEDS_CONTEXT cases: Missing contract must prevent confident judgment.
For INTENTIONAL cases: The behavior must be deliberate design."""

        user_prompt = f"""Create a complete benchmark test case.

Case Plan:
- Title: {plan.title}
- Language: {plan.language}
- Technical Domain: {plan.technical_domain}
- Primary Concept: {plan.primary_concept}
- Failure Mechanism: {plan.failure_mechanism}
- Expected Conclusion: {plan.expected_conclusion.value}
- Difficulty: {plan.difficulty.value}
- Code Shape: {plan.code_shape}
- Semantic Signature: {plan.semantic_signature}

Generate the complete markdown benchmark prompt including:
1. Standard review instructions header
2. Any necessary context/contract
3. A realistic code sample in {plan.language}

The code should test: {plan.case_summary}

Output the complete markdown content only."""

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        
        return self.client.extract_content(response)

    def _validate_markdown(self, markdown: str) -> bool:
        """Validate that markdown has required structure."""
        # Must have a code block
        if "```" not in markdown:
            return False
        
        # Should not contain obvious answer leaks
        forbidden_patterns = [
            "this code contains a bug",
            "this code is safe",
            "this code is correct",
            "BUG:",
            "DEFECT:",
            "the problem here",
        ]
        
        lower_md = markdown.lower()
        for pattern in forbidden_patterns:
            if pattern in lower_md:
                return False
        
        return True

    def _extract_ground_truth(self, plan: CasePlan, markdown: str) -> str:
        """Generate ground truth explanation for private storage."""
        base = f"This case tests {plan.primary_concept}. "
        
        if plan.expected_conclusion == ExpectedConclusion.BUG:
            base += f"The defect is: {plan.failure_mechanism}."
        elif plan.expected_conclusion == ExpectedConclusion.CORRECT:
            base += "The code correctly handles the suspected issue."
        elif plan.expected_conclusion == ExpectedConclusion.UNSUPPORTED:
            base += "The candidate issue depends on assumptions not in evidence."
        elif plan.expected_conclusion == ExpectedConclusion.NEEDS_CONTEXT:
            base += "Missing contract/context prevents confident defect claim."
        else:  # INTENTIONAL
            base += "The behavior is intentional by design."
        
        return base

    def _describe_evidence(self, plan: CasePlan, markdown: str) -> str:
        """Describe what evidence establishes the intended judgment."""
        if plan.expected_conclusion == ExpectedConclusion.BUG:
            return f"Visible code shows {plan.failure_mechanism} without mitigation."
        elif plan.expected_conclusion == ExpectedConclusion.CORRECT:
            return "Code includes explicit handling that addresses the suspected issue."
        elif plan.expected_conclusion == ExpectedConclusion.UNSUPPORTED:
            return "No visible evidence supports the candidate defect claims."
        elif plan.expected_conclusion == ExpectedConclusion.NEEDS_CONTEXT:
            return "Contract ambiguity or missing context creates uncertainty."
        else:
            return "Design intent is clear from code structure and patterns."
