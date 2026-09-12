"""Diversity validation and similarity checking."""

import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional

from .models import CasePlan, GeneratedCase


# Initial "already used" concept catalog from existing benchmarks
INITIAL_CONCEPTS = frozenset([
    "canonicalized value is computed but discarded before enqueue/use",
    "immutable update is correctly captured and used",
    "write failure incorrectly reports success",
    "swallowed read/IO failure leaves invalid/null state",
    "resource/stream is not closed",
    "null/blank guard correctly protects later access",
    "parameterized SQL query is already safe",
    "explicit sealed success/failure result is correctly handled",
    "guard/bounds check occurs after dangerous access",
    "correct try/finally behavior",
    "local-only counter with no demonstrated race",
    "intentional error-handling behavior that should not be reported as a defect",
])


_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})([^\r\n]*)$")
_FENCE_CLOSE_RE = re.compile(r"^[ \t]{0,3}([`~]{3,})[ \t]*$")


def extract_code_from_markdown(markdown: str | None) -> Optional[str]:
    """Return the first non-empty, properly closed Markdown code fence.

    The info string is intentionally opaque: language names such as ``c++``
    and ``c#`` are valid without maintaining a language allow-list.  A fence
    must be closed by the same marker character with at least the opening
    marker length; inline backticks and stray markers do not qualify.
    """

    if not isinstance(markdown, str):
        return None

    lines = markdown.splitlines(keepends=True)
    for index, line in enumerate(lines):
        opening = _FENCE_OPEN_RE.match(line.rstrip("\r\n"))
        if opening is None:
            continue

        marker = opening.group(1)
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            closing = _FENCE_CLOSE_RE.match(candidate.rstrip("\r\n"))
            if (
                closing is not None
                and closing.group(1)[0] == marker[0]
                and len(closing.group(1)) >= len(marker)
            ):
                code = "".join(body)
                return code if code.strip() else None
            body.append(candidate)

        # Preserve first-fence semantics: once an opening fence is seen, an
        # unterminated or empty first block is invalid rather than falling
        # through to a later block.
        return None

    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_tokens(text: str) -> set[str]:
    """Extract normalized tokens from text."""
    text = normalize_text(text)
    # Remove common programming punctuation
    text = re.sub(r"[{}()\[\];:,.\-+=*/&|^~<>!?@#$%]", " ", text)
    tokens = text.split()
    # Filter very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return set(tokens)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def sequence_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher similarity ratio."""
    return SequenceMatcher(None, a, b).ratio()


def compute_code_similarity(code1: str, code2: str) -> float:
    """Compute similarity between two code samples."""
    norm1 = normalize_text(code1)
    norm2 = normalize_text(code2)
    
    # Token-based similarity
    tokens1 = extract_tokens(norm1)
    tokens2 = extract_tokens(norm2)
    token_sim = jaccard_similarity(tokens1, tokens2)
    
    # Sequence-based similarity
    seq_sim = sequence_similarity(norm1, norm2)
    
    # Weighted combination
    return 0.4 * token_sim + 0.6 * seq_sim


def concepts_collide(
    primary_concept1: str,
    failure_mechanism1: str,
    semantic_sig1: str,
    primary_concept2: str,
    failure_mechanism2: str,
    semantic_sig2: str,
) -> tuple[bool, str]:
    """
    Check if two cases have colliding concepts.
    
    Returns (collision_detected, reason_if_any).
    """
    pc1 = normalize_text(primary_concept1)
    pc2 = normalize_text(primary_concept2)
    fm1 = normalize_text(failure_mechanism1)
    fm2 = normalize_text(failure_mechanism2)
    ss1 = normalize_text(semantic_sig1)
    ss2 = normalize_text(semantic_sig2)
    
    # Exact match on primary concept
    if pc1 == pc2:
        return True, f"Primary concept '{primary_concept1}' duplicates existing concept"
    
    # Very similar primary concepts
    if sequence_similarity(pc1, pc2) > 0.85:
        return True, f"Primary concept too similar: '{primary_concept1}' vs '{primary_concept2}'"
    
    # Same failure mechanism with similar concept
    if fm1 == fm2 and sequence_similarity(pc1, pc2) > 0.6:
        return True, f"Same failure mechanism with similar concept"
    
    # Semantic signature collision
    if ss1 and ss2 and sequence_similarity(ss1, ss2) > 0.9:
        return True, f"Semantic signatures are substantially equivalent"
    
    return False, ""


class DiversityTracker:
    """Tracks diversity across generated cases and validates new candidates."""

    def __init__(self):
        self._cases: list[GeneratedCase] = []
        self._concepts: set[str] = set(INITIAL_CONCEPTS)
        self._mechanisms: set[str] = set()
        self._signatures: list[tuple[str, str]] = []  # (normalized_concept, normalized_signature)

    @property
    def case_count(self) -> int:
        return len(self._cases)

    def load_cases(self, cases: list[GeneratedCase]) -> None:
        """Load existing cases into the tracker."""
        for case in cases:
            self._cases.append(case)
            self._concepts.add(normalize_text(case.plan.primary_concept))
            self._mechanisms.add(normalize_text(case.plan.failure_mechanism))
            self._signatures.append((
                normalize_text(case.plan.primary_concept),
                normalize_text(case.plan.semantic_signature),
            ))

    def validate_plan(
        self,
        plan: CasePlan,
        code_sample: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Validate a case plan against existing cases.
        
        Returns (is_valid, rejection_reason_if_any).
        """
        # Check against initial concepts
        normalized_concept = normalize_text(plan.primary_concept)
        for existing in self._concepts:
            if normalized_concept == existing:
                return False, f"Concept '{plan.primary_concept}' matches occupied concept '{existing}'"
            if sequence_similarity(normalized_concept, existing) > 0.85:
                return False, f"Concept too similar to existing: '{existing}'"
        
        # Check against all existing cases
        for case in self._cases:
            collide, reason = concepts_collide(
                plan.primary_concept,
                plan.failure_mechanism,
                plan.semantic_signature,
                case.plan.primary_concept,
                case.plan.failure_mechanism,
                case.plan.semantic_signature,
            )
            if collide:
                return False, reason
        
        # Check code similarity if provided
        if code_sample and self._cases:
            for case in self._cases:
                # Extract code from markdown if available
                existing_code = self._extract_code_from_markdown(case.markdown_content)
                if existing_code:
                    sim = compute_code_similarity(code_sample, existing_code)
                    if sim > 0.75:
                        return False, f"Code sample too similar to existing case {case.filename} (similarity: {sim:.2f})"
        
        return True, ""

    def _extract_code_from_markdown(self, markdown: str) -> str:
        """Extract code block content from markdown."""
        return extract_code_from_markdown(markdown) or ""

    def add_case(self, case: GeneratedCase) -> None:
        """Add an accepted case to the tracker."""
        self._cases.append(case)
        self._concepts.add(normalize_text(case.plan.primary_concept))
        self._mechanisms.add(normalize_text(case.plan.failure_mechanism))
        self._signatures.append((
            normalize_text(case.plan.primary_concept),
            normalize_text(case.plan.semantic_signature),
        ))

    def get_existing_concepts_summary(self) -> str:
        """Get a summary of existing concepts for model feedback."""
        if not self._cases:
            return "No cases generated yet."
        
        lines = ["Existing primary concepts:"]
        for i, case in enumerate(self._cases[-10:], start=max(1, len(self._cases) - 9)):
            lines.append(f"  {i}. {case.plan.primary_concept} ({case.plan.language})")
        return "\n".join(lines)
