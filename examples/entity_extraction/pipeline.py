"""Pipeline example: entity extraction as a pipeline component.

Shows how to use TonicTextualEntityExtractor in a Haystack pipeline,
chained with other components.

Usage:
    cd examples/entity_extraction
    uv run pipeline.py
"""

from haystack import Pipeline
from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)

pipeline = Pipeline()
pipeline.add_component("extractor", TonicTextualEntityExtractor())

documents = [
    Document(content="Patient Maria Garcia, DOB 03/15/1982, MRN 12345678."),
    Document(content="Invoice for Acme Corp, attn: Bob Jones, bob@acme.com."),
]

result = pipeline.run({"extractor": {"documents": documents}})

for doc in result["extractor"]["documents"]:
    entities = TonicTextualEntityExtractor.get_stored_annotations(doc)
    print(f"\nDocument: {doc.content[:60]}...")
    for entity in entities:
        print(f"  {entity.entity}: {entity.text}")
