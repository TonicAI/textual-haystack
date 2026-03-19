"""Integration tests for TonicTextualDocumentCleaner.

These tests verify that the component can talk to the Tonic Textual API
and returns structurally valid results. They deliberately avoid asserting
that specific PII was or wasn't detected, since the underlying NER model
may change.
"""

from __future__ import annotations

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
)


@pytest.mark.integration
def test_redaction_returns_string() -> None:
    cleaner = TonicTextualDocumentCleaner(generator_default="Redaction")
    result = cleaner.run(
        documents=[
            Document(content="My name is John Smith and my email is john@example.com.")
        ]
    )

    cleaned = result["documents"][0].content
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


@pytest.mark.integration
def test_synthesis_returns_string() -> None:
    cleaner = TonicTextualDocumentCleaner(generator_default="Synthesis")
    result = cleaner.run(
        documents=[
            Document(content="My name is John Smith and my email is john@example.com.")
        ]
    )

    cleaned = result["documents"][0].content
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


@pytest.mark.integration
def test_per_entity_config_accepted() -> None:
    cleaner = TonicTextualDocumentCleaner(
        generator_default="Off",
        generator_config={
            "NAME_GIVEN": "Redaction",
            "EMAIL_ADDRESS": "Synthesis",
        },
    )
    result = cleaner.run(
        documents=[
            Document(content="Contact John at john@example.com about the project.")
        ]
    )

    cleaned = result["documents"][0].content
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


@pytest.mark.integration
def test_multiple_documents() -> None:
    cleaner = TonicTextualDocumentCleaner(generator_default="Redaction")
    result = cleaner.run(
        documents=[
            Document(content="Jane Doe, jane@example.com"),
            Document(content="Bob Jones, (555) 867-5309"),
        ]
    )

    docs = result["documents"]
    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0


@pytest.mark.integration
def test_does_not_mutate_original() -> None:
    original = Document(
        content="John Smith at john@example.com",
        meta={"source": "test"},
    )
    cleaner = TonicTextualDocumentCleaner(generator_default="Redaction")
    result = cleaner.run(documents=[original])

    assert original.content == "John Smith at john@example.com"
    assert result["documents"][0].meta["source"] == "test"
