"""Unit tests for TonicTextualDocumentCleaner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
)


@dataclass
class _MockRedactionResponse:
    redacted_text: str


def _make_cleaner(
    **kwargs: Any,
) -> TonicTextualDocumentCleaner:
    """Create a cleaner with a mocked client."""
    kwargs.setdefault("api_key", Secret.from_token("fake-key"))
    cleaner = TonicTextualDocumentCleaner(**kwargs)
    cleaner._client = MagicMock()
    return cleaner


def test_redaction_default() -> None:
    cleaner = _make_cleaner()
    assert cleaner._client is not None

    cleaner._client.redact.return_value = _MockRedactionResponse(
        redacted_text="My name is [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx]"
    )

    docs = [Document(content="My name is John Smith")]
    result = cleaner.run(documents=docs)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == (
        "My name is [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx]"
    )
    cleaner._client.redact.assert_called_once_with(
        "My name is John Smith",
        generator_default="Redaction",
    )


def test_synthesis_mode() -> None:
    cleaner = _make_cleaner(generator_default="Synthesis")
    assert cleaner._client is not None

    cleaner._client.redact.return_value = _MockRedactionResponse(
        redacted_text="My name is Maria Chen"
    )

    docs = [Document(content="My name is John Smith")]
    result = cleaner.run(documents=docs)

    assert result["documents"][0].content == "My name is Maria Chen"
    cleaner._client.redact.assert_called_once_with(
        "My name is John Smith",
        generator_default="Synthesis",
    )


def test_generator_config() -> None:
    cleaner = _make_cleaner(
        generator_default="Off",
        generator_config={
            "NAME_GIVEN": "Synthesis",
            "NAME_FAMILY": "Synthesis",
            "EMAIL_ADDRESS": "Redaction",
        },
    )
    assert cleaner._client is not None

    cleaner._client.redact.return_value = _MockRedactionResponse(
        redacted_text="Maria Chen at [EMAIL_ADDRESS_xxxx]"
    )

    docs = [Document(content="John Smith at john@example.com")]
    result = cleaner.run(documents=docs)

    assert result["documents"][0].content == "Maria Chen at [EMAIL_ADDRESS_xxxx]"
    cleaner._client.redact.assert_called_once_with(
        "John Smith at john@example.com",
        generator_default="Off",
        generator_config={
            "NAME_GIVEN": "Synthesis",
            "NAME_FAMILY": "Synthesis",
            "EMAIL_ADDRESS": "Redaction",
        },
    )


def test_skip_none_content() -> None:
    cleaner = _make_cleaner()
    assert cleaner._client is not None

    docs = [Document(content=None)]
    result = cleaner.run(documents=docs)
    assert len(result["documents"]) == 1
    cleaner._client.redact.assert_not_called()


def test_no_mutation_of_input() -> None:
    cleaner = _make_cleaner()
    assert cleaner._client is not None

    cleaner._client.redact.return_value = _MockRedactionResponse(
        redacted_text="[NAME_GIVEN_xxxx]"
    )

    original = Document(content="John", meta={"existing_key": "value"})
    result = cleaner.run(documents=[original])

    assert original.content == "John"
    result_doc = result["documents"][0]
    assert result_doc.content == "[NAME_GIVEN_xxxx]"
    assert result_doc.meta["existing_key"] == "value"


def test_multiple_documents() -> None:
    cleaner = _make_cleaner()
    assert cleaner._client is not None

    cleaner._client.redact.side_effect = [
        _MockRedactionResponse(redacted_text="[NAME_GIVEN_xxxx]"),
        _MockRedactionResponse(redacted_text="[EMAIL_ADDRESS_xxxx]"),
    ]

    docs = [
        Document(content="John"),
        Document(content="john@example.com"),
    ]
    result = cleaner.run(documents=docs)
    assert len(result["documents"]) == 2
    assert result["documents"][0].content == "[NAME_GIVEN_xxxx]"
    assert result["documents"][1].content == "[EMAIL_ADDRESS_xxxx]"


def test_api_error_passes_through_original() -> None:
    cleaner = _make_cleaner()
    assert cleaner._client is not None

    cleaner._client.redact.side_effect = RuntimeError("API error")

    docs = [Document(content="John Smith")]
    result = cleaner.run(documents=docs)
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "John Smith"


def test_serialization_round_trip(monkeypatch: Any) -> None:
    monkeypatch.setenv("TONIC_TEXTUAL_API_KEY", "test-key")
    cleaner = TonicTextualDocumentCleaner(
        base_url="https://textual.example.com",
        generator_default="Synthesis",
        generator_config={"NAME_GIVEN": "Synthesis", "EMAIL_ADDRESS": "Redaction"},
    )
    data = cleaner.to_dict()

    assert data["init_parameters"]["base_url"] == "https://textual.example.com"
    assert data["init_parameters"]["generator_default"] == "Synthesis"
    assert data["init_parameters"]["generator_config"] == {
        "NAME_GIVEN": "Synthesis",
        "EMAIL_ADDRESS": "Redaction",
    }

    restored = TonicTextualDocumentCleaner.from_dict(data)
    assert restored.base_url == "https://textual.example.com"
    assert restored.generator_default == "Synthesis"
    assert restored.generator_config == {
        "NAME_GIVEN": "Synthesis",
        "EMAIL_ADDRESS": "Redaction",
    }


def test_warm_up_only_once() -> None:
    with patch(
        "haystack_integrations.components.tonic_textual.document_cleaner.TextualNer"
    ) as mock_ner:
        cleaner = TonicTextualDocumentCleaner(api_key=Secret.from_token("fake-key"))
        cleaner.warm_up()
        cleaner.warm_up()
        mock_ner.assert_called_once()


def test_empty_generator_config_not_passed() -> None:
    cleaner = _make_cleaner(generator_config={})
    assert cleaner._client is not None

    cleaner._client.redact.return_value = _MockRedactionResponse(
        redacted_text="[NAME_GIVEN_xxxx]"
    )

    cleaner.run(documents=[Document(content="John")])
    cleaner._client.redact.assert_called_once_with(
        "John",
        generator_default="Redaction",
    )
