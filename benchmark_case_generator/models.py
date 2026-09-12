"""Models for benchmark case generation."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Literal
from enum import Enum


class ExpectedConclusion(str, Enum):
    """Possible expected conclusions for a benchmark case."""
    BUG = "BUG"
    CORRECT = "CORRECT"
    UNSUPPORTED = "UNSUPPORTED"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    INTENTIONAL = "INTENTIONAL"


class Difficulty(str, Enum):
    """Difficulty levels for benchmark cases."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class CasePlan:
    """Structured plan for a single benchmark case."""
    title: str
    language: str
    technical_domain: str
    primary_concept: str
    failure_mechanism: str
    expected_conclusion: ExpectedConclusion
    difficulty: Difficulty
    code_shape: str
    semantic_signature: str
    case_summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "language": self.language,
            "technical_domain": self.technical_domain,
            "primary_concept": self.primary_concept,
            "failure_mechanism": self.failure_mechanism,
            "expected_conclusion": self.expected_conclusion.value,
            "difficulty": self.difficulty.value,
            "code_shape": self.code_shape,
            "semantic_signature": self.semantic_signature,
            "case_summary": self.case_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CasePlan":
        """Create from dictionary."""
        if not isinstance(data, Mapping):
            raise ValueError("case plan must be a JSON object")

        text_fields = (
            "title",
            "language",
            "technical_domain",
            "primary_concept",
            "failure_mechanism",
            "expected_conclusion",
            "difficulty",
            "code_shape",
            "semantic_signature",
            "case_summary",
        )
        for field_name in text_fields:
            value = data.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"case plan field '{field_name}' must be non-empty text"
                )

        try:
            expected_conclusion = ExpectedConclusion(data["expected_conclusion"])
            difficulty = Difficulty(data["difficulty"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("case plan contains an invalid enum value") from error

        return cls(
            title=data["title"],
            language=data["language"],
            technical_domain=data["technical_domain"],
            primary_concept=data["primary_concept"],
            failure_mechanism=data["failure_mechanism"],
            expected_conclusion=expected_conclusion,
            difficulty=difficulty,
            code_shape=data["code_shape"],
            semantic_signature=data["semantic_signature"],
            case_summary=data["case_summary"],
        )


@dataclass
class GeneratedCase:
    """A fully generated benchmark case with metadata."""
    filename: str
    # ``None`` is reserved for legacy state files written before Markdown was
    # persisted. New generated cases always carry the complete logical text.
    markdown_content: str | None
    markdown_sha256: str
    plan: CasePlan
    ground_truth_explanation: str
    evidence_description: str
    timestamp: str
    model_identifier: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "markdown_content": self.markdown_content,
            "markdown_sha256": self.markdown_sha256,
            "title": self.plan.title,
            "language": self.plan.language,
            "technical_domain": self.plan.technical_domain,
            "primary_concept": self.plan.primary_concept,
            "failure_mechanism": self.plan.failure_mechanism,
            "expected_conclusion": self.plan.expected_conclusion.value,
            "difficulty": self.plan.difficulty.value,
            "semantic_signature": self.plan.semantic_signature,
            "ground_truth_explanation": self.ground_truth_explanation,
            "evidence_description": self.evidence_description,
            "timestamp": self.timestamp,
            "model_identifier": self.model_identifier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedCase":
        """Create from dictionary."""
        plan = CasePlan(
            title=data["title"],
            language=data["language"],
            technical_domain=data["technical_domain"],
            primary_concept=data["primary_concept"],
            failure_mechanism=data["failure_mechanism"],
            expected_conclusion=ExpectedConclusion(data["expected_conclusion"]),
            difficulty=Difficulty(data["difficulty"]),
            code_shape=data.get("code_shape", ""),
            semantic_signature=data["semantic_signature"],
            case_summary="",
        )
        markdown_content = data.get("markdown_content")
        if markdown_content is not None and not isinstance(markdown_content, str):
            raise ValueError("markdown_content must be text when present")
        return cls(
            filename=data["filename"],
            markdown_content=markdown_content,
            markdown_sha256=data["markdown_sha256"],
            plan=plan,
            ground_truth_explanation=data["ground_truth_explanation"],
            evidence_description=data["evidence_description"],
            timestamp=data["timestamp"],
            model_identifier=data["model_identifier"],
        )
