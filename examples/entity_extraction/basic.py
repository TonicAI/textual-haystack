"""Basic entity extraction example.

Extract PII entities from documents and inspect the results.

Usage:
    cd examples/entity_extraction
    uv run basic.py
"""

from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)

documents = [
    Document(
        content=(
            "My name is John Smith and my email is john@example.com. "
            "I live at 123 Main St, Springfield, IL 62704. "
            "My SSN is 123-45-6789."
        )
    ),
    Document(
        content="Contact Jane Doe at (555) 867-5309 or jane.doe@acme.com."
    ),
]

extractor = TonicTextualEntityExtractor()
result = extractor.run(documents=documents)

for i, doc in enumerate(result["documents"]):
    entities = TonicTextualEntityExtractor.get_stored_annotations(doc)
    print(f"\n--- Document {i + 1} ({len(entities)} entities) ---")
    for entity in entities:
        print(
            f"  {entity.entity:25s} "
            f"{entity.text:30s} "
            f"[{entity.start}:{entity.end}] "
            f"(score: {entity.score:.2f})"
        )
