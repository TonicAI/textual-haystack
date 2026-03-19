# textual-haystack

[![PyPI version](https://img.shields.io/pypi/v/textual-haystack)](https://pypi.org/project/textual-haystack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/tonicai/textual-haystack/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

PII detection, transformation, and entity extraction components for [Haystack](https://haystack.deepset.ai/), powered by [Tonic Textual](https://textual.tonic.ai).

Detect sensitive data in documents, extract the raw entities for auditing or custom logic, or synthesize and tokenize PII before ingestion. Drop these components into any Haystack pipeline.

## Installation

```bash
pip install textual-haystack
```

## Components

| Component | Purpose |
|-----------|---------|
| `TonicTextualEntityExtractor` | Extract PII entities with type, value, location, and confidence score |
| `TonicTextualDocumentCleaner` | Synthesize or tokenize PII in document content (coming soon) |

## Quick start

```bash
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

### Entity extraction

```python
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)

extractor = TonicTextualEntityExtractor()
result = extractor.run(
    documents=[Document(content="My name is John Smith and my email is john@example.com")]
)

for entity in TonicTextualEntityExtractor.get_stored_annotations(result["documents"][0]):
    print(f"{entity.entity}: {entity.text} (confidence: {entity.score:.2f})")
# NAME_GIVEN: John (confidence: 0.90)
# NAME_FAMILY: Smith (confidence: 0.90)
# EMAIL_ADDRESS: john@example.com (confidence: 0.95)
```

### In a pipeline

```python
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import (
    TonicTextualEntityExtractor,
)

pipeline = Pipeline()
pipeline.add_component("extractor", TonicTextualEntityExtractor())

result = pipeline.run({
    "extractor": {
        "documents": [
            Document(content="Contact Jane Doe at jane@example.com"),
        ]
    }
})

for doc in result["extractor"]["documents"]:
    entities = TonicTextualEntityExtractor.get_stored_annotations(doc)
    print(f"Found {len(entities)} entities")
```

## Configuration

**Self-hosted deployment:**

```python
extractor = TonicTextualEntityExtractor(
    base_url="https://textual.your-company.com"
)
```

**Explicit API key:**

```python
from haystack.utils.auth import Secret

extractor = TonicTextualEntityExtractor(
    api_key=Secret.from_token("your-api-key")
)
```

## Development

```bash
# install dependencies
uv sync --group dev --group test --group lint --group typing

# run unit tests
make test

# run integration tests (requires TONIC_TEXTUAL_API_KEY)
make integration_tests

# lint & format
make lint
make format
```

## License

MIT
