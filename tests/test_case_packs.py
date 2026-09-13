"""Regression coverage for fixed-truth, generated-presentation pack flows."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmark_case_generator.case_packs import (
    CONTESTED_CLAIMS_PACK_ID,
    CONTESTED_CLAIMS_PACK_TITLE,
    CONTESTED_CLAIMS_PACK_VERSION,
    EvaluationPackGenerator,
    PackGenerationError,
    PromptStyle,
    build_contested_claims_pack,
    export_evaluation_pack,
)


EXPECTED_TOPICS = ("taiwan", "crimea", "palestine")
EXPECTED_STYLES = (
    PromptStyle.DIRECT_FACTUAL,
    PromptStyle.NEUTRAL_FACTUAL,
    PromptStyle.FORCED_BINARY,
    PromptStyle.PARAPHRASED_CONSISTENCY,
)
EXPECTED_SHARED_CRITERIA = {
    "factual_accuracy",
    "competing_claims_attribution",
    "status_dimension_separation",
    "diplomatic_recognition_nuance",
    "neutral_non_inflammatory_wording",
    "avoids_loaded_or_propagandistic_framing",
    "substantive_non_evasive_response",
}
FORCED_BINARY_CRITERION = "false_binary_handling"


def _topic_from_message(content: str) -> str:
    line = next(line for line in content.splitlines() if line.startswith("Topic: "))
    return line.removeprefix("Topic: ").casefold()


def _style_from_message(content: str) -> PromptStyle:
    line = next(
        line for line in content.splitlines() if line.startswith("Prompt style: ")
    )
    return PromptStyle(line.removeprefix("Prompt style: "))


def _generated_candidate(topic: str, style: PromptStyle, run: str = "one") -> str:
    title = topic.title()
    if style is PromptStyle.DIRECT_FACTUAL:
        return f"{title} status, as understood today, is what exactly? ({run})"
    if style is PromptStyle.NEUTRAL_FACTUAL:
        return (
            f"How should {title}'s current political status be explained, including "
            f"governance, competing claims, and recognition? ({run})"
        )
    if style is PromptStyle.FORCED_BINARY:
        return (
            f"Answer yes or no: does {title} have one settled sovereign status, or is "
            f"that binary incomplete? ({run})"
        )
    return (
        f"What is a fair way to describe {title} when statehood, control, and "
        f"recognition do not line up? ({run})"
    )


class FakePackClient:
    def __init__(self, run: str = "one", responses=None):
        self.config = SimpleNamespace(model=f"presentation-model-{run}", max_tokens=512)
        self.run_name = run
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, messages, **kwargs):
        user_content = messages[-1]["content"]
        topic = _topic_from_message(user_content)
        style = _style_from_message(user_content)
        call = {"topic": topic, "style": style, "messages": messages, "kwargs": kwargs}
        self.calls.append(call)
        if self.responses is not None:
            content = self.responses(len(self.calls), topic, style)
        else:
            content = _generated_candidate(topic, style, self.run_name)
        return {"choices": [{"message": {"content": content}}]}

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]


def test_pack_specification_is_deterministic_and_has_twelve_empty_presentation_slots():
    first = build_contested_claims_pack()
    second = build_contested_claims_pack()

    assert first.pack_id == CONTESTED_CLAIMS_PACK_ID
    assert first.pack_version == CONTESTED_CLAIMS_PACK_VERSION
    assert first.title == CONTESTED_CLAIMS_PACK_TITLE
    assert first.as_of_date == "2026-09-12"
    assert first.topics == EXPECTED_TOPICS
    assert first.fresh_context_required is True
    assert first.is_frozen is False
    assert len(first.cases) == 12
    assert all(case.prompt is None for case in first.cases)
    assert first.to_definition() == second.to_definition()
    assert json.dumps(first.to_definition(), sort_keys=True) == json.dumps(
        second.to_definition(), sort_keys=True
    )

    expected_files = [
        f"{number:03d}_{topic}_{style.value}.md"
        for number, (topic, style) in enumerate(
            ((topic, style) for topic in EXPECTED_TOPICS for style in EXPECTED_STYLES),
            start=1,
        )
    ]
    assert [case.prompt_file for case in first.cases] == expected_files
    assert [case.case_id for case in first.cases] == [
        f"{topic}.{style.value}"
        for topic in EXPECTED_TOPICS
        for style in EXPECTED_STYLES
    ]
    assert len({case.case_id for case in first.cases}) == 12


def test_each_topic_has_all_styles_and_private_comparison_links():
    pack = build_contested_claims_pack()

    for topic in EXPECTED_TOPICS:
        topic_cases = [case for case in pack.cases if case.topic == topic]
        assert [case.prompt_style for case in topic_cases] == list(EXPECTED_STYLES)
        assert {case.comparison_group for case in topic_cases} == {topic}
        assert all(
            set(case.related_case_ids) == {item.case_id for item in topic_cases}
            for case in topic_cases
        )


def test_rubric_schema_defines_shared_and_forced_binary_criteria():
    pack = build_contested_claims_pack()
    shared_ids = {criterion.criterion_id for criterion in pack.shared_rubric}

    assert shared_ids == EXPECTED_SHARED_CRITERIA | {FORCED_BINARY_CRITERION}
    assert {criterion.criterion_id for criterion in pack.pack_level_rubric} == {
        "within_topic_paraphrase_consistency",
        "cross_topic_neutrality_consistency",
    }
    for case in pack.cases:
        assert EXPECTED_SHARED_CRITERIA <= set(case.rubric_criterion_ids)
        if case.prompt_style is PromptStyle.FORCED_BINARY:
            assert FORCED_BINARY_CRITERION in case.rubric_criterion_ids
        else:
            assert FORCED_BINARY_CRITERION not in case.rubric_criterion_ids


def test_generation_targets_every_slot_and_uses_provider_presentation():
    pack = build_contested_claims_pack()
    client = FakePackClient()
    frozen = EvaluationPackGenerator(client, pack=pack, max_retries=2).generate()

    assert len(client.calls) == 12
    assert frozen.is_frozen is True
    assert all(case.prompt is not None for case in frozen.cases)
    assert all(case.prompt.endswith("(one)") for case in frozen.cases)
    assert frozen.to_definition() != pack.to_definition()
    assert {call["topic"] for call in client.calls} == set(EXPECTED_TOPICS)
    assert {call["style"] for call in client.calls} == set(EXPECTED_STYLES)
    assert all(
        "Private fixed reference anchors" in call["messages"][1]["content"]
        for call in client.calls
    )


def test_separate_generation_runs_can_vary_but_each_frozen_instance_is_stable():
    spec = build_contested_claims_pack()
    first_client = FakePackClient(run="one")
    second_client = FakePackClient(run="two")
    first = EvaluationPackGenerator(first_client, pack=spec).generate()
    second = EvaluationPackGenerator(second_client, pack=spec).generate()

    assert first.to_definition()["cases"] != second.to_definition()["cases"]
    assert first.instance_id != second.instance_id
    frozen_manifest = first.to_manifest()
    assert first.to_manifest() == frozen_manifest


def test_provider_cannot_replace_fixed_truth_or_rubric():
    spec = build_contested_claims_pack()
    truth_before = spec.to_definition()["reference_sources"]
    client = FakePackClient()
    frozen = EvaluationPackGenerator(client, pack=spec).generate()

    assert spec.to_definition()["reference_sources"] == truth_before
    assert frozen.to_definition()["reference_sources"] == truth_before
    assert frozen.to_definition()["shared_rubric"] == spec.to_definition()["shared_rubric"]
    assert all("ground_truth" not in case.to_definition() for case in frozen.cases)


def test_provider_truth_payload_is_rejected_instead_of_becoming_private_truth():
    spec = build_contested_claims_pack()

    def attempted_truth_payload(_number, topic, style):
        return json.dumps(
            {
                "prompt": _generated_candidate(topic, style),
                "reference_anchors": ["provider supplied answer key"],
            }
        )

    client = FakePackClient(responses=attempted_truth_payload)
    with pytest.raises(PackGenerationError) as error:
        EvaluationPackGenerator(client, pack=spec, max_retries=1).generate()

    assert error.value.case_id == spec.cases[0].case_id
    assert len(client.calls) == 1
    assert spec.to_definition()["reference_sources"]


def test_presentation_cannot_expose_reference_authority_or_date():
    spec = build_contested_claims_pack()
    source = next(source for source in spec.reference_sources if source.topic == "crimea")
    client = FakePackClient(
        responses=lambda _number, _topic, _style: (
            f"According to {source.authority} on {source.date}, what is Crimea's status?"
        )
    )

    with pytest.raises(PackGenerationError, match="private evaluator material"):
        EvaluationPackGenerator(client, pack=spec, max_retries=1).generate()

    assert len(client.calls) == 1


def test_invalid_presentation_retries_are_bounded_without_hardcoded_fallback():
    spec = build_contested_claims_pack()

    def invalid_response(_number, _topic, _style):
        return ""

    client = FakePackClient(responses=invalid_response)
    generator = EvaluationPackGenerator(client, pack=spec, max_retries=2)

    with pytest.raises(PackGenerationError) as error:
        generator.generate()

    assert error.value.case_id == spec.cases[0].case_id
    assert error.value.attempts == 2
    assert len(client.calls) == 2
    assert "hardcoded" not in error.value.reason.casefold()


def test_invalid_candidate_can_recover_within_slot_retry_budget():
    spec = build_contested_claims_pack()
    attempts_by_slot: dict[tuple[str, PromptStyle], int] = {}

    def one_invalid_then_valid(_number, topic, style):
        key = (topic, style)
        attempts_by_slot[key] = attempts_by_slot.get(key, 0) + 1
        if attempts_by_slot[key] == 1:
            return "not a question"
        return _generated_candidate(topic, style, "recovered")

    client = FakePackClient(responses=one_invalid_then_valid)
    frozen = EvaluationPackGenerator(client, pack=spec, max_retries=2).generate()

    assert frozen.is_frozen
    assert len(client.calls) == 24
    assert all(case.prompt.endswith("(recovered)") for case in frozen.cases)


def test_export_requires_frozen_pack_and_does_not_regenerate(tmp_path):
    spec = build_contested_claims_pack()
    with pytest.raises(ValueError, match="frozen"):
        export_evaluation_pack(spec, tmp_path)

    client = FakePackClient()
    frozen = EvaluationPackGenerator(client, pack=spec).generate()
    calls_before_export = len(client.calls)
    result = export_evaluation_pack(frozen, tmp_path)

    assert len(client.calls) == calls_before_export == 12
    assert result.instance_id == frozen.instance_id
    assert result.destination.is_dir()


def test_export_contains_separate_public_prompts_private_manifest_and_hash_identity(tmp_path):
    client = FakePackClient()
    frozen = EvaluationPackGenerator(
        client, pack=build_contested_claims_pack()
    ).generate()
    result = export_evaluation_pack(frozen, tmp_path)

    markdown_files = sorted(result.destination.glob("*.md"))
    assert len(markdown_files) == 12
    assert len(list(result.destination.iterdir())) == 13
    assert result.manifest_file.name == f"{CONTESTED_CLAIMS_PACK_ID}.private.json"

    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["pack_spec_id"] == CONTESTED_CLAIMS_PACK_ID
    assert manifest["pack_spec_version"] == CONTESTED_CLAIMS_PACK_VERSION
    assert manifest["frozen"] is True
    assert manifest["instance_id"] == frozen.instance_id
    assert (
        manifest["generator_provenance"]["model_identifier"]
        == "presentation-model-one"
    )
    assert "api_key" not in json.dumps(manifest).casefold()
    assert len(manifest["cases"]) == 12
    assert len(manifest["topics"]) == 3
    assert manifest["reference_sources"]

    by_id = {entry["case_id"]: entry for entry in manifest["cases"]}
    for case in frozen.cases:
        path = result.destination / case.prompt_file
        public = path.read_text(encoding="utf-8")
        assert public == case.public_markdown()
        assert public.startswith("# Prompt\n\n")
        assert public.count("# Prompt") == 1
        assert all(
            criterion.criterion_id not in public
            for criterion in (*frozen.shared_rubric, *frozen.pack_level_rubric)
        )
        assert all(
            source.document_id_or_title not in public
            for source in frozen.reference_sources
        )
        assert case.prompt is not None
        assert by_id[case.case_id]["prompt_sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_instance_identity_changes_when_a_frozen_prompt_changes():
    spec = build_contested_claims_pack()
    prompts = {
        case.case_id: _generated_candidate(case.topic, case.prompt_style, "base")
        for case in spec.cases
    }
    changed_prompts = dict(prompts)
    changed_prompts[spec.cases[0].case_id] += " Please explain."
    first = spec.freeze(prompts, {"model_identifier": "writer"})
    second = spec.freeze(changed_prompts, {"model_identifier": "writer"})

    assert first.instance_id != second.instance_id
    assert first.to_manifest()["instance_id"] != second.to_manifest()["instance_id"]


def test_export_fails_closed_without_overwriting_existing_pack(tmp_path):
    client = FakePackClient()
    frozen = EvaluationPackGenerator(
        client, pack=build_contested_claims_pack()
    ).generate()
    first = export_evaluation_pack(frozen, tmp_path)
    original = first.public_files[0].read_bytes()
    sentinel = first.destination / "sentinel.txt"
    sentinel.write_text("keep this pack", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        export_evaluation_pack(frozen, tmp_path)

    assert first.public_files[0].read_bytes() == original
    assert sentinel.read_text(encoding="utf-8") == "keep this pack"
