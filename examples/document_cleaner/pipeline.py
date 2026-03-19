"""Pipeline example: document cleaner before ingestion.

Shows how to use TonicTextualDocumentCleaner in a Haystack pipeline
to clean PII from documents before they reach downstream components.

Usage:
    cd examples/document_cleaner
    uv run pipeline.py
"""

from haystack import Pipeline
from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
    TonicTextualEntityExtractor,
)

# Build a pipeline that first cleans PII, then extracts any remaining entities
# to verify the cleaning worked.
pipeline = Pipeline()
pipeline.add_component(
    "cleaner",
    TonicTextualDocumentCleaner(generator_default="Synthesis"),
)
pipeline.add_component("extractor", TonicTextualEntityExtractor())
pipeline.connect("cleaner", "extractor")

documents = [
    Document(content="Contact Jane Doe at jane@example.com or (555) 123-4567."),
    Document(content="Patient Bob Jones, SSN 987-65-4321, MRN 00112233."),
]

result = pipeline.run({"cleaner": {"documents": documents}})

for i, doc in enumerate(result["extractor"]["documents"]):
    entities = TonicTextualEntityExtractor.get_stored_annotations(doc)
    print(f"\n--- Document {i + 1} ---")
    print(f"Cleaned: {doc.content}")
    print(f"Entities remaining: {len(entities)}")
    for entity in entities:
        print(f"  {entity.entity}: {entity.text}")
