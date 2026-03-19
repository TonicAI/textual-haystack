# Entity Extraction Examples

Extract PII entities from documents using Tonic Textual and Haystack.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A [Tonic Textual](https://textual.tonic.ai) API key

## Setup

```bash
cd examples/entity_extraction
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

## Examples

### Basic usage

Extract entities from documents and inspect the results:

```bash
uv run basic.py
```

### Pipeline usage

Use the extractor as a component in a Haystack pipeline:

```bash
uv run pipeline.py
```
