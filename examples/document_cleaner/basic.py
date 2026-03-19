"""Basic document cleaner examples.

Shows synthesis, tokenization, and per-entity control.

Usage:
    cd examples/document_cleaner
    uv run basic.py
"""

from haystack.dataclasses import Document

from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
)

docs = [
    Document(
        content=(
            "Patient John Smith, DOB 03/15/1982, SSN 123-45-6789. "
            "Contact at john.smith@example.com or (555) 867-5309."
        )
    ),
]

# --- Synthesis: replace PII with realistic fakes ---
print("=== Synthesis ===")
cleaner = TonicTextualDocumentCleaner(generator_default="Synthesis")
result = cleaner.run(documents=docs)
print(result["documents"][0].content)

# --- Tokenization: replace PII with reversible placeholders ---
print("\n=== Tokenization ===")
cleaner = TonicTextualDocumentCleaner(generator_default="Redaction")
result = cleaner.run(documents=docs)
print(result["documents"][0].content)

# --- Per-entity control: mix modes per PII type ---
print("\n=== Per-entity control ===")
cleaner = TonicTextualDocumentCleaner(
    generator_default="Off",
    generator_config={
        "NAME_GIVEN": "Synthesis",
        "NAME_FAMILY": "Synthesis",
        "EMAIL_ADDRESS": "Redaction",
        "US_SSN": "Redaction",
        "PHONE_NUMBER": "Redaction",
    },
)
result = cleaner.run(documents=docs)
print(result["documents"][0].content)
