"""Integration tests for TonicTextualEntityExtractor.

These tests verify that the component can talk to the Tonic Textual API
and returns structurally valid results. They deliberately avoid asserting
specific entity labels or counts, since the underlying NER model may
change what it detects.
"""

from __future__ import annotations

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)
from haystack_integrations.components.tonic_textual.entity_extractor import (
    PiiEntityAnnotation,
)


@pytest.mark.integration
def test_extract_returns_valid_structure() -> None:
    extractor = TonicTextualEntityExtractor()
    result = extractor.run(
        documents=[
            Document(content="My name is John Smith and my email is john@example.com.")
        ]
    )

    docs = result["documents"]
    assert len(docs) == 1
    annotations = TonicTextualEntityExtractor.get_stored_annotations(docs[0])
    assert isinstance(annotations, list)
    assert all(isinstance(a, PiiEntityAnnotation) for a in annotations)


@pytest.mark.integration
def test_extract_multiple_documents() -> None:
    extractor = TonicTextualEntityExtractor()
    result = extractor.run(
        documents=[
            Document(content="Call Jane Doe at (555) 867-5309."),
            Document(content="Order #12345 shipped to 123 Main St."),
        ]
    )

    docs = result["documents"]
    assert len(docs) == 2
    for doc in docs:
        annotations = TonicTextualEntityExtractor.get_stored_annotations(doc)
        assert isinstance(annotations, list)


@pytest.mark.integration
def test_extract_preserves_content() -> None:
    original_text = "Contact Bob Jones at bob@example.com."
    extractor = TonicTextualEntityExtractor()
    result = extractor.run(documents=[Document(content=original_text)])

    assert result["documents"][0].content == original_text


@pytest.mark.integration
def test_extract_annotation_fields_well_formed() -> None:
    extractor = TonicTextualEntityExtractor()
    result = extractor.run(
        documents=[Document(content="Email me at test@example.com.")]
    )

    annotations = TonicTextualEntityExtractor.get_stored_annotations(
        result["documents"][0]
    )
    for a in annotations:
        assert isinstance(a.entity, str) and len(a.entity) > 0
        assert isinstance(a.text, str) and len(a.text) > 0
        assert isinstance(a.start, int) and a.start >= 0
        assert isinstance(a.end, int) and a.end > a.start
        assert isinstance(a.score, float)


@pytest.mark.integration
def test_extract_preserves_existing_meta() -> None:
    extractor = TonicTextualEntityExtractor()
    result = extractor.run(
        documents=[Document(content="John Smith", meta={"source": "test", "page": 1})]
    )

    doc = result["documents"][0]
    assert doc.meta["source"] == "test"
    assert doc.meta["page"] == 1
