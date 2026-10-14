# Architecture

```mermaid
flowchart LR
    A[CSV upload / pasted text / REST API] --> B[Validation]
    B --> C[Preprocessing<br/>normalise, PII strip, typo repair]
    C --> D[Classifier<br/>TF-IDF + Logistic Regression<br/>multi-label]
    C --> E[Entity extraction<br/>spaCy NER + rules]
    E --> F[Location resolver<br/>gazetteer, optional geocoder]
    C --> G[Duplicate detection<br/>Sentence Transformers or offline fallback]
    D --> H[(SQLite)]
    E --> H
    F --> H
    G --> H
    H --> I[Analytics]
    H --> J[FastAPI]
    H --> K[Streamlit dashboard]
    I --> K
    K --> L[Human review<br/>confirm / correct / reject]
    L --> H
```

```
                +----------------------------+
 Reports -----> |  nlp/pipeline.py           |  analyze()  ingest()  ingest_dataframe()
 (CSV, text,    |   preprocessing            |
  REST)         |   classifier   entities    |      +--------------+
                |   locations    duplicates  | ---> | SQLite       |  messages, entities,
                +----------------------------+      | database/    |  classifications,
                       ^                |           +--------------+  duplicate_suggestions,
                       |                |                  |          reviews, model_evaluations
        artifacts/ (model, metrics)     |                  v
                                        |        analytics.py (shared)
                                        v                  |
                         api/main.py (FastAPI)       app.py (Streamlit, 9 pages)
```

Principles: original text is never modified; every extracted field carries evidence; unknowns stay "Unknown";
duplicates are suggestions only; humans can correct labels and locations; the core app needs no internet.

The dashboard calls the pipeline directly (no HTTP hop) so it keeps working if the API is not running;
both share the same SQLite file.


## v2: ingestion layer
```mermaid
flowchart LR
    S1[NDMA Sachet RSS + CAP] --> A[ingestion adapters<br/>parse, validate, normalise]
    S2[GDACS GeoJSON] --> A
    S3[ReliefWeb API v2] --> A
    A --> N[NormalizedItem]
    N --> P[Pipeline.process_external<br/>entities, summary, classification where valid]
    P --> DB[(SQLite messages + provenance)]
    SV[IngestionService<br/>lease lock, intervals, back-off, quota] --> A
    SCH[scheduler / manual button] --> SV
    SV --> RUNS[(ingestion_runs, ingestion_errors, source_state)]
    DB --> UI[Streamlit pages + FastAPI]
```
Imported items live in the existing `messages` table (new provenance columns), so review, duplicates, NLP analysis and analytics keep working.
