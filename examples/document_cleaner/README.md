# Document Cleaner Examples

Clean PII from documents using Tonic Textual and Haystack.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A [Tonic Textual](https://textual.tonic.ai) API key

## Setup

```bash
cd examples/document_cleaner
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

## Examples

### Basic usage

Shows synthesis, tokenization, and per-entity control:

```bash
uv run basic.py
```

### Pipeline usage

Cleans documents then runs entity extraction to verify the cleaning:

```bash
uv run pipeline.py
```
