# Announcing textual-haystack: PII-Safe Document Processing for Haystack Pipelines

Every RAG pipeline starts with ingestion. Documents come in — PDFs, support tickets, clinical notes, customer records — get chunked, embedded, and stored in a vector database. From that point on, every query that retrieves those chunks surfaces whatever was in the original documents. If the source data contains names, email addresses, Social Security numbers, or medical record numbers, that PII propagates through the entire system: into the embeddings, into the retrieval results, and into the LLM's context window.

This creates two problems that are hard to solve after the fact:

**The first is compliance.** Once PII is embedded in a vector store, you can't easily remove it. GDPR right-to-erasure requests, HIPAA de-identification requirements, PCI cardholder data rules — these all require that sensitive data be handled before it becomes entangled in your retrieval infrastructure. Redacting a source document after its chunks are already embedded doesn't fix the embeddings. You need to clean the data before ingestion.

**The second is retrieval quality.** Dense embeddings encode everything in the text, including PII that's irrelevant to the semantic content you actually want to retrieve on. A clinical note about a specific diagnosis will produce different embeddings depending on which patient name appears in it. Entity extraction gives you structured metadata — names, organizations, locations, dates — that you can use to build hybrid retrieval strategies, filter results, or enrich chunks with faceted metadata that improves precision without relying solely on semantic similarity.

These aren't niche concerns. They show up in every organization building RAG over sensitive data:

- **Healthcare RAG** over clinical notes, discharge summaries, and lab reports — where HIPAA requires de-identification before the data touches any system that isn't covered under the same BAA.
- **Legal document search** over contracts, correspondence, and case files — where client names and financial details need to be scrubbed before indexing, but entity metadata (parties, dates, jurisdictions) improves retrieval.
- **Customer support knowledge bases** built from ticket histories — where customer PII needs to be removed but the semantic content of the resolution needs to be preserved.
- **Financial services** building search over transaction records, compliance reports, and client communications — where PCI and privacy regulations constrain what can be stored and retrieved.

Regex and rule-based approaches can catch an email that looks like `user@example.com`, but they fall apart on context-dependent PII. Is "Jordan" a person's name or a country? Is "April" a name or a month? Is "1600 Pennsylvania Avenue" a location that needs redaction or a well-known reference? You need NER models that understand context, not pattern matching.

[Tonic Textual](https://tonic.ai/textual) provides exactly this — transformer-based PII detection and transformation across text, JSON, HTML, PDFs, images, and tabular data. With `textual-haystack`, those capabilities are now available as native Haystack pipeline components.

## What Tonic Textual does

Textual's NER model identifies 46+ entity types across 50+ languages. These aren't just the obvious ones like email addresses and phone numbers. The model detects names (given and family, separately), dates of birth, occupations, healthcare IDs, routing numbers, IP addresses, and more — with a confidence score for each detection.

What happens after detection is where Textual differentiates itself. There are three capabilities:

**Synthesis** replaces PII with realistic fake data that preserves the structure and statistical properties of the original:

```
Input:  "Patient John Smith, DOB 03/15/1982, MRN 12345678"
Output: "Patient Maria Chen, DOB 07/22/1975, MRN 87654321"
```

Synthesized data keeps downstream analytics, embeddings, and retrieval valid. A synthesized name is still a plausible name in the right position in the sentence. A synthesized date has the right format. The semantic structure of the document is preserved while the identifying information is replaced. This matters for RAG — placeholder tokens like `[NAME_GIVEN_xxxx]` distort the embeddings your retriever learns from, while synthesized replacements preserve the natural language distribution.

**Tokenization** replaces detected PII with labeled placeholders:

```
Input:  "Patient John Smith, DOB 03/15/1982, MRN 12345678"
Output: "Patient [NAME_GIVEN_a1b2] [NAME_FAMILY_c3d4], DOB [DOB_e5f6], MRN [HEALTHCARE_ID_g7h8]"
```

The placeholders are tagged with their entity type and a consistent identifier, so you can track which replacements correspond to the same original entity across a document. This is the safest option when you need to guarantee that no real PII appears in the output.

**Entity extraction** returns the raw detections without modifying the text:

```
Input:  "Patient John Smith, DOB 03/15/1982, MRN 12345678"
Output: [
  {"entity": "NAME_GIVEN", "text": "John", "start": 8, "end": 12, "score": 0.95},
  {"entity": "NAME_FAMILY", "text": "Smith", "start": 13, "end": 18, "score": 0.95},
  {"entity": "DOB", "text": "03/15/1982", "start": 24, "end": 34, "score": 0.90},
  {"entity": "HEALTHCARE_ID", "text": "12345678", "start": 40, "end": 48, "score": 0.85}
]
```

Entity extraction is useful when you need to know *what* PII is present — for auditing, for building structured metadata that improves retrieval, or for making per-document decisions about how to handle the data.

## The Haystack integration

`textual-haystack` provides two Haystack components:

| Component | Purpose |
|-----------|---------|
| `TonicTextualDocumentCleaner` | Synthesize or tokenize PII in document content before ingestion |
| `TonicTextualEntityExtractor` | Extract PII entities and store them as document metadata |

Both are native Haystack `@component` classes. They accept `list[Document]` as input and return `list[Document]` as output, so they slot directly into any Haystack pipeline. Installation is a single line:

```bash
pip install textual-haystack
```

### TonicTextualDocumentCleaner

The document cleaner transforms PII in document content before it reaches downstream components like splitters, embedders, and document stores. It produces new `Document` instances with cleaned content — the originals are never mutated.

```python
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import TonicTextualDocumentCleaner

cleaner = TonicTextualDocumentCleaner(generator_default="Synthesis")
result = cleaner.run(documents=[
    Document(content="Patient John Smith, DOB 03/15/1982, was admitted for chest pain.")
])
print(result["documents"][0].content)
# "Patient Maria Chen, DOB 07/22/1975, was admitted for chest pain."
```

The clinical content — "admitted for chest pain" — is preserved. The identifying information is replaced with synthetic data that maintains the same structure. When this document is chunked and embedded, the resulting vectors capture the medical semantics without encoding any real patient data.

#### Per-entity control

In practice, you often want different handling for different entity types. Names might be safe to synthesize, but SSNs should always be tokenized. Organization names might be left alone entirely.

`generator_config` provides this control:

```python
cleaner = TonicTextualDocumentCleaner(
    generator_default="Off",
    generator_config={
        "NAME_GIVEN": "Synthesis",
        "NAME_FAMILY": "Synthesis",
        "DOB": "Synthesis",
        "US_SSN": "Redaction",
        "EMAIL_ADDRESS": "Redaction",
    },
)
```

This configuration synthesizes names and dates of birth (preserving natural language flow for better embeddings), tokenizes SSNs and emails (for maximum safety), and leaves everything else untouched. The `generator_default` of `"Off"` means any entity type not listed in `generator_config` passes through unchanged.

### TonicTextualEntityExtractor

The entity extractor detects PII in document content and stores the results as structured metadata in `doc.meta["named_entities"]`. The document content itself is not modified.

```python
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import TonicTextualEntityExtractor

extractor = TonicTextualEntityExtractor()
result = extractor.run(documents=[
    Document(content="Contact Jane Doe at jane@example.com or (555) 867-5309.")
])

for entity in TonicTextualEntityExtractor.get_stored_annotations(result["documents"][0]):
    print(f"{entity.entity}: {entity.text} (confidence: {entity.score:.2f})")
# NAME_GIVEN: Jane (confidence: 0.95)
# NAME_FAMILY: Doe (confidence: 0.95)
# EMAIL_ADDRESS: jane@example.com (confidence: 0.90)
# PHONE_NUMBER: (555) 867-5309 (confidence: 0.90)
```

Each annotation is a `PiiEntityAnnotation` dataclass with `entity` (the PII type label), `text` (the detected value), `start` and `end` (character offsets), and `score` (confidence).

#### Why entity extraction improves retrieval

Dense retrieval works well for semantic similarity, but it struggles with precision when you need to find documents about a specific person, organization, or date. Entity extraction gives you structured metadata you can use for:

- **Hybrid retrieval** — combine dense vector search with metadata filters. Retrieve documents semantically similar to the query, then filter by extracted entities (e.g., "find documents about billing disputes" filtered to a specific customer or date range).
- **Faceted search** — expose extracted entities as facets in a search UI, letting users narrow results by person, organization, location, or date.
- **Chunk enrichment** — attach entity metadata to chunks before embedding, so retrieval-time filters can operate at the chunk level.
- **Audit and compliance** — know exactly which documents contain which types of PII, without manually reviewing every document.

## Putting it together: a pipeline

Here's a complete pipeline that cleans documents and then extracts entities from the cleaned text — a common pattern when you want both PII-safe content and structured metadata:

```python
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
    TonicTextualEntityExtractor,
)

pipeline = Pipeline()
pipeline.add_component(
    "cleaner",
    TonicTextualDocumentCleaner(generator_default="Synthesis"),
)
pipeline.add_component("extractor", TonicTextualEntityExtractor())
pipeline.connect("cleaner", "extractor")

documents = [
    Document(content="Patient John Smith, DOB 03/15/1982, MRN 12345678."),
    Document(content="Invoice for Acme Corp, attn: Bob Jones, bob@acme.com."),
]

result = pipeline.run({"cleaner": {"documents": documents}})

for doc in result["extractor"]["documents"]:
    entities = TonicTextualEntityExtractor.get_stored_annotations(doc)
    print(f"\nCleaned: {doc.content}")
    print(f"Entities: {[(e.entity, e.text) for e in entities]}")
```

The cleaner runs first, replacing real PII with synthesized data. The extractor then runs on the cleaned text, producing entity metadata from the synthetic values. The result is documents with PII-safe content *and* structured entity metadata — ready for chunking, embedding, and storage.

You can also use either component independently. The cleaner alone is sufficient for ingestion pipelines where you just need PII-safe content. The extractor alone is useful when you want entity metadata without modifying the source documents — for example, in an analytics pipeline where you need to catalog what PII exists across a document corpus.

## How both components work under the hood

Both components follow Haystack's conventions:

- **`@component` decorated** — no base class inheritance, just the decorator and a `run()` method.
- **Input/output**: `run(documents: list[Document]) -> {"documents": list[Document]}`.
- **No mutation**: both create new `Document` instances rather than modifying inputs. The cleaner uses `dataclasses.replace()` to produce documents with transformed content. The extractor uses `replace()` to produce documents with enriched metadata.
- **Lazy client initialization**: the `warm_up()` method initializes the Tonic Textual client once, on first use.
- **Pipeline serialization**: both implement `to_dict()` and `from_dict()` using Haystack's `default_to_dict`/`default_from_dict`, with `Secret` for API key handling.

The Tonic Textual client is shared across calls within a component instance. The API key is read from the `TONIC_TEXTUAL_API_KEY` environment variable by default, or can be passed explicitly via Haystack's `Secret`:

```python
from haystack.utils.auth import Secret

cleaner = TonicTextualDocumentCleaner(
    api_key=Secret.from_token("your-api-key"),
    base_url="https://textual.your-company.com",  # for self-hosted deployments
)
```

## What's next

This initial release covers the two highest-impact use cases for Haystack pipelines: document cleaning before ingestion and entity extraction for metadata enrichment. We're continuing to expand capabilities.

The package is open source under the MIT license:

- **PyPI:** [textual-haystack](https://pypi.org/project/textual-haystack/)
- **GitHub:** [tonicai/textual-haystack](https://github.com/tonicai/textual-haystack)

Self-contained examples for both components live in the `examples/` directory of the repo.

To get started:

```bash
pip install textual-haystack
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

```python
from haystack.dataclasses import Document
from haystack_integrations.components.tonic_textual import (
    TonicTextualDocumentCleaner,
    TonicTextualEntityExtractor,
)

# Clean PII before ingestion
cleaner = TonicTextualDocumentCleaner(generator_default="Synthesis")

# Or extract entities for metadata enrichment
extractor = TonicTextualEntityExtractor()
```

For the full Tonic Textual platform — including the web UI, dataset management, custom entity training, and enterprise deployment — visit [tonic.ai/textual](https://tonic.ai/textual).
