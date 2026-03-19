"""Unit tests for TonicTextualEntityExtractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)
from haystack_integrations.components.tonic_textual.entity_extractor import (
    PiiEntityAnnotation,
)


@dataclass
class _MockReplacement:
    label: str
    text: str
    start: int
    end: int
    score: float


@dataclass
class _MockRedactionResponse:
    redacted_text: str
    de_identify_results: list[_MockReplacement]


def _make_extractor() -> TonicTextualEntityExtractor:
    """Create an extractor with a mocked client."""
    extractor = TonicTextualEntityExtractor(api_key=Secret.from_token("fake-key"))
    extractor._client = MagicMock()
    return extractor


def test_extract_entities() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    extractor._client.redact.return_value = _MockRedactionResponse(
        redacted_text="My name is [NAME_GIVEN_xxxx]",
        de_identify_results=[
            _MockReplacement(
                label="NAME_GIVEN", text="John", start=11, end=15, score=0.95
            ),
        ],
    )

    docs = [Document(content="My name is John")]
    result = extractor.run(documents=docs)

    assert len(result["documents"]) == 1
    doc = result["documents"][0]
    assert doc.content == "My name is John"  # content unchanged

    annotations = TonicTextualEntityExtractor.get_stored_annotations(doc)
    assert len(annotations) == 1
    assert annotations[0].entity == "NAME_GIVEN"
    assert annotations[0].text == "John"
    assert annotations[0].start == 11
    assert annotations[0].end == 15
    assert annotations[0].score == 0.95


def test_extract_multiple_entities() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    extractor._client.redact.return_value = _MockRedactionResponse(
        redacted_text="...",
        de_identify_results=[
            _MockReplacement(
                label="NAME_GIVEN", text="John", start=11, end=15, score=0.9
            ),
            _MockReplacement(
                label="NAME_FAMILY", text="Smith", start=16, end=21, score=0.9
            ),
            _MockReplacement(
                label="EMAIL_ADDRESS",
                text="john@example.com",
                start=38,
                end=54,
                score=0.95,
            ),
        ],
    )

    docs = [Document(content="My name is John Smith and my email is john@example.com")]
    result = extractor.run(documents=docs)
    annotations = TonicTextualEntityExtractor.get_stored_annotations(
        result["documents"][0]
    )
    assert len(annotations) == 3
    labels = [a.entity for a in annotations]
    assert labels == ["NAME_GIVEN", "NAME_FAMILY", "EMAIL_ADDRESS"]


def test_skip_none_content() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    docs = [Document(content=None)]
    result = extractor.run(documents=docs)
    assert len(result["documents"]) == 1
    assert (
        TonicTextualEntityExtractor.get_stored_annotations(result["documents"][0]) == []
    )
    extractor._client.redact.assert_not_called()


def test_no_mutation_of_input() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    extractor._client.redact.return_value = _MockRedactionResponse(
        redacted_text="...",
        de_identify_results=[
            _MockReplacement(
                label="NAME_GIVEN", text="John", start=0, end=4, score=0.9
            ),
        ],
    )

    original = Document(content="John", meta={"existing_key": "value"})
    result = extractor.run(documents=[original])

    # Original doc should not have named_entities
    assert "named_entities" not in original.meta
    # Result doc should have both existing meta and new annotations
    result_doc = result["documents"][0]
    assert result_doc.meta["existing_key"] == "value"
    assert len(result_doc.meta["named_entities"]) == 1


def test_multiple_documents() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    extractor._client.redact.side_effect = [
        _MockRedactionResponse(
            redacted_text="...",
            de_identify_results=[
                _MockReplacement(
                    label="NAME_GIVEN", text="John", start=0, end=4, score=0.9
                ),
            ],
        ),
        _MockRedactionResponse(
            redacted_text="...",
            de_identify_results=[
                _MockReplacement(
                    label="EMAIL_ADDRESS",
                    text="jane@example.com",
                    start=0,
                    end=16,
                    score=0.95,
                ),
            ],
        ),
    ]

    docs = [
        Document(content="John"),
        Document(content="jane@example.com"),
    ]
    result = extractor.run(documents=docs)
    assert len(result["documents"]) == 2
    assert (
        TonicTextualEntityExtractor.get_stored_annotations(result["documents"][0])[
            0
        ].entity
        == "NAME_GIVEN"
    )
    assert (
        TonicTextualEntityExtractor.get_stored_annotations(result["documents"][1])[
            0
        ].entity
        == "EMAIL_ADDRESS"
    )


def test_api_error_returns_empty_annotations() -> None:
    extractor = _make_extractor()
    assert extractor._client is not None

    extractor._client.redact.side_effect = RuntimeError("API error")

    docs = [Document(content="John Smith")]
    result = extractor.run(documents=docs)
    assert len(result["documents"]) == 1
    annotations = TonicTextualEntityExtractor.get_stored_annotations(
        result["documents"][0]
    )
    assert annotations == []


def test_serialization_round_trip(monkeypatch: Any) -> None:
    monkeypatch.setenv("TONIC_TEXTUAL_API_KEY", "test-key")
    extractor = TonicTextualEntityExtractor(
        base_url="https://textual.example.com",
    )
    data = extractor.to_dict()

    assert data["init_parameters"]["base_url"] == "https://textual.example.com"
    assert "api_key" in data["init_parameters"]

    restored = TonicTextualEntityExtractor.from_dict(data)
    assert restored.base_url == "https://textual.example.com"


def test_warm_up_only_once() -> None:
    with patch(
        "haystack_integrations.components.tonic_textual.entity_extractor.TextualNer"
    ) as mock_ner:
        extractor = TonicTextualEntityExtractor(api_key=Secret.from_token("fake-key"))
        extractor.warm_up()
        extractor.warm_up()
        mock_ner.assert_called_once()


def test_get_stored_annotations_missing() -> None:
    doc = Document(content="hello")
    assert TonicTextualEntityExtractor.get_stored_annotations(doc) == []


def test_pii_entity_annotation_fields() -> None:
    annotation = PiiEntityAnnotation(
        entity="EMAIL_ADDRESS",
        text="john@example.com",
        start=0,
        end=16,
        score=0.95,
    )
    assert annotation.entity == "EMAIL_ADDRESS"
    assert annotation.text == "john@example.com"
    assert annotation.start == 0
    assert annotation.end == 16
    assert annotation.score == 0.95
