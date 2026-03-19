"""PII entity extraction component using Tonic Textual."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils.auth import Secret

from tonic_textual.redact_api import TextualNer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class PiiEntityAnnotation:
    """A single PII entity detected in a document.

    Attributes:
        entity: The PII entity type label (e.g. ``NAME_GIVEN``, ``EMAIL_ADDRESS``).
        text: The actual text that was detected.
        start: Start character offset in the document content.
        end: End character offset in the document content (exclusive).
        score: Confidence score of the detection.
    """

    entity: str
    text: str
    start: int
    end: int
    score: float


@component
class TonicTextualEntityExtractor:
    """Extract PII entities from documents using Tonic Textual.

    Detects personally identifiable information in document content and
    stores the results as annotations in ``doc.meta["named_entities"]``.
    The document content is not modified.

    Usage:

    ```python
    from haystack_integrations.components.tonic_textual import (
        TonicTextualEntityExtractor,
    )

    extractor = TonicTextualEntityExtractor()
    result = extractor.run(
        documents=[Document(content="My name is John Smith, email john@example.com")]
    )
    entities = TonicTextualEntityExtractor.get_stored_annotations(
        result["documents"][0]
    )
    for entity in entities:
        print(f"{entity.entity}: {entity.text} ({entity.score:.2f})")
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TONIC_TEXTUAL_API_KEY"),  # noqa: B008
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._client: TextualNer | None = None

    def warm_up(self) -> None:
        """Initialize the Tonic Textual client."""
        if self._client is not None:
            return
        kwargs: dict[str, Any] = {
            "api_key": self.api_key.resolve_value() or "",
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = TextualNer(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Extract PII entities from each document.

        Args:
            documents: Documents to extract entities from.

        Returns:
            A dictionary with key ``"documents"`` containing the input
            documents enriched with entity annotations in
            ``doc.meta["named_entities"]``.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101

        result: list[Document] = []
        for doc in documents:
            if doc.content is None:
                logger.warning(
                    "Document '%s' has no text content, skipping entity extraction.",
                    doc.id,
                )
                result.append(doc)
                continue

            try:
                response = self._client.redact(doc.content)
                annotations = [
                    PiiEntityAnnotation(
                        entity=r.label,
                        text=r.text,
                        start=r.start,
                        end=r.end,
                        score=r.score,
                    )
                    for r in response.de_identify_results
                ]
            except Exception:
                logger.exception(
                    "Failed to extract entities from document '%s'.", doc.id
                )
                annotations = []

            new_meta = {**doc.meta, "named_entities": annotations}
            result.append(replace(doc, meta=new_meta))

        return {"documents": result}

    @staticmethod
    def get_stored_annotations(document: Document) -> list[PiiEntityAnnotation]:
        """Retrieve entity annotations previously stored on a document.

        Args:
            document: A document that has been processed by this component.

        Returns:
            A list of ``PiiEntityAnnotation`` objects, or an empty list if
            no annotations are present.
        """
        return document.meta.get("named_entities", [])

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component for pipeline export."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            base_url=self.base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TonicTextualEntityExtractor:
        """Deserialize this component from a pipeline export."""
        init_params = data.get("init_parameters", {})
        if "api_key" in init_params:
            init_params["api_key"] = Secret.from_dict(init_params["api_key"])
        return default_from_dict(cls, data)
