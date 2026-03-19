"""PII document cleaner component using Tonic Textual."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from tonic_textual.redact_api import TextualNer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@component
class TonicTextualDocumentCleaner:
    """Clean PII from documents using Tonic Textual.

    Detects personally identifiable information in document content and
    replaces it via synthesis (realistic fake data) or tokenization
    (reversible placeholders). The cleaned content is written to new
    ``Document`` instances; the originals are never mutated.

    Usage:

    ```python
    from haystack_integrations.components.tonic_textual import (
        TonicTextualDocumentCleaner,
    )

    # Synthesize PII with realistic fakes
    cleaner = TonicTextualDocumentCleaner(generator_default="Synthesis")
    result = cleaner.run(
        documents=[Document(content="My name is John Smith, email john@example.com")]
    )
    print(result["documents"][0].content)
    # "My name is Maria Chen, email maria.chen@gmail.com"
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TONIC_TEXTUAL_API_KEY"),  # noqa: B008
        base_url: str | None = None,
        generator_default: Literal["Off", "Redaction", "Synthesis"] = "Redaction",
        generator_config: dict[str, Literal["Off", "Redaction", "Synthesis"]]
        | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.generator_default = generator_default
        self.generator_config = generator_config or {}
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

    def _build_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the Textual API call."""
        kwargs: dict[str, Any] = {
            "generator_default": self.generator_default,
        }
        if self.generator_config:
            kwargs["generator_config"] = self.generator_config
        return kwargs

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Clean PII from each document.

        Args:
            documents: Documents to clean.

        Returns:
            A dictionary with key ``"documents"`` containing new
            ``Document`` instances with PII-transformed content.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101

        result: list[Document] = []
        for doc in documents:
            if doc.content is None:
                logger.warning(
                    "Document '%s' has no text content, skipping.",
                    doc.id,
                )
                result.append(doc)
                continue

            try:
                response = self._client.redact(doc.content, **self._build_kwargs())
                cleaned = replace(doc, content=response.redacted_text)
            except Exception:
                logger.exception("Failed to clean document '%s'.", doc.id)
                cleaned = doc

            result.append(cleaned)

        return {"documents": result}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component for pipeline export."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            base_url=self.base_url,
            generator_default=self.generator_default,
            generator_config=self.generator_config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TonicTextualDocumentCleaner:
        """Deserialize this component from a pipeline export."""
        init_params = data.get("init_parameters", {})
        if "api_key" in init_params:
            init_params["api_key"] = Secret.from_dict(init_params["api_key"])
        return default_from_dict(cls, data)
