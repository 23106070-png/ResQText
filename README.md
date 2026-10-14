# ResQText: Disaster Message Intelligence

NLP decision-support for emergency teams. It reads incoming disaster reports, classifies the needs, extracts locations / people / resources with the supporting text, flags possible duplicates, shows trends, and puts every result in front of a human reviewer.

> **Decision support only.** Not connected to emergency services, not a replacement for official systems, and never dispatches anything. All bundled demo data is **synthetic**.

## Version 2: real-world disaster intelligence
ResQText can now import **official alerts and reports** and show them next to submitted messages. Everything from v1 still works.

| Source | Needs | Notes |
|---|---|---|
| NDMA Sachet (India alerts) | nothing | public CAP/RSS feeds, ETag caching as NDMA requires |
| GDACS (global events) | nothing | credit "Global Disaster Awareness and Coordination System, GDACS" |
| ReliefWeb (situation reports) | **an approved `appname` you must request** | skipped automatically until set |

Details, rules and what is still untested: **docs/DATA_SOURCES.md**.

**New pages:** Real-World Updates, Disaster Event Map, Data Sources and Health. Overview, Emerging Needs, NLP Analysis and Human Review were upgraded.
**New API endpoints:** `GET /reports`, `GET /reports/{id}`, `GET /events`, `GET /sources/status`, `POST /ingestion/refresh`, `GET /ingestion/runs`, `POST /reports/{id}/apply-update`, `GET /analytics/real-world`. Write endpoints need `X-API-Key` when `RESQTEXT_API_KEY` is set and are **disabled in production until it is**.

### Run it (Windows PowerShell)
Run these **inside the project folder** (the one containing `app.py`), in a normal PowerShell window (not Administrator):
```powershell
cd C:\Users\YOU\Downloads\ResQText
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                     # no new packages in v2
copy .env.example .env                              # then edit .env (e.g. RELIEFWEB_APPNAME=...)
python -m training.train_classifier                 # only if artifacts\ is empty
python -m ingestion status                          # which sources are configured / working
python -m ingestion refresh --force                 # fetch now from the command line
streamlit run app.py                                # dashboard  http://localhost:8501
uvicorn api.main:app                                # API        http://127.0.0.1:8000/docs
python -m ingestion.scheduler                       # OPTIONAL third terminal: automatic polling
pytest -q
```
Database upgrade is automatic: opening an existing `resqtext.db` adds the new columns/tables and keeps all data.

### Safety and honesty rules built in
* An official alert is never turned into a "need". Official records are not classified.
* Real reports are always flagged **Needs human review**: the classifier is trained on synthetic data only.
* Reviewed items are never overwritten by a newer source version without your click.
* Sources show **Connected** only after a recorded successful fetch; **Stale** and **Failing** are shown plainly. This is periodic polling, not a real-time feed.
* Write actions fail closed in `RESQTEXT_ENV=production` until a passcode / API key is configured. Error details are visible to signed-in reviewers only.
* Event-map markers are approximate centroids and never show a person's location.
* Do not put `streamlit run` / `uvicorn --reload` straight on the internet: use HTTPS via a reverse proxy.

## What is inside
| Area | Implementation |
|---|---|
| Classification | TF-IDF (word + character n-grams) + Logistic Regression, multi-label, per-label scores. Optional zero-shot transformer (off by default). |
| Entities | spaCy NER + custom rules: locations, people counts, resources, dates/times, emergency keywords, organisations. Each with an evidence span. |
| Locations | Offline gazetteer (approximate centres), ambiguity detection, manual correction, optional Nominatim geocoding (off). Never invents coordinates. |
| Duplicates | Sentence Transformers when enabled; offline hashed n-gram fallback otherwise. Location and time aware. Human confirms or rejects. |
| Analytics | Category counts, volume over time, location and resource rankings, window-over-window change with an honest caveat. |
| Dashboard | Streamlit, 12 pages in 4 groups, navy/white design. |
| API | FastAPI + Swagger UI. |
| Storage | SQLite, parameterised queries. |

## Setup (Windows PowerShell)
Needs Python 3.10 to 3.12.
```powershell
cd ResQText
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m training.synthetic_data       # optional: regenerate the synthetic sample data
python -m training.prepare_dataset      # clean, normalise, split, leakage check
python -m training.train_classifier     # train + evaluate -> artifacts\
```
Run (two terminals, both with the venv active):
```powershell
streamlit run app.py                    # dashboard  -> http://localhost:8501
uvicorn api.main:app --reload           # API        -> http://127.0.0.1:8000/docs
```
The dashboard does not need the API running. Tests: `pytest -q`

Optional extras:
```powershell
pip install sentence-transformers
$env:RESQTEXT_USE_SENTENCE_TRANSFORMERS = "1"     # downloads all-MiniLM-L6-v2 on first use
$env:RESQTEXT_ENABLE_GEOCODING = "1"              # online lookups, 1 request/second
```
Or copy `.env.example` to `.env`.

## Using the dashboard
1. **Data and Settings**: click *Load simulated demo reports* (trains first if needed).
2. **Overview**: counts, categories, review queue (low confidence and unclear locations first).
3. **NLP Analysis**: paste a message, see categories with scores, highlighted evidence and a structured table.
4. **Incoming Messages**: filter reports; upload a CSV (template provided).
5. **Duplicate Detection**: compare pairs side by side, confirm or reject.
6. **Location Map / Emerging Needs**: map of report clusters, trends and change between time windows.
7. **Human Review**: correct categories and location, approve or reject, export reviewed results to CSV.
8. **Model Evaluation**: metrics calculated from the held-out test set.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, model/spaCy/embedder state |
| POST | `/messages/analyze` | Analyse one report (`save: true` also stores it) |
| POST | `/messages/batch` | Analyse and store up to 1000 reports |
| GET | `/messages` | List with filters: `category`, `location`, `review_status`, `min_confidence`, `limit`, `offset` |
| GET | `/messages/{id}` | Report with entities, scores, latest review |
| GET | `/messages/{id}/duplicates` | Duplicate suggestions |
| POST | `/duplicates/{id}/decision` | Confirm / reject a suggestion |
| PATCH | `/messages/{id}/location` | Manual location correction |
| GET | `/analytics/summary` | Aggregates and caveat |
| POST | `/reviews` | Record a human review |
| GET | `/evaluation` | Latest evaluation report |

```powershell
curl.exe -X POST http://127.0.0.1:8000/messages/analyze -H "Content-Type: application/json" `
  -d '{\"text\": \"We are trapped near the municipal school in Kalwa. Five people need drinking water.\"}'
```
Errors: 422 invalid input, 404 unknown id, 503 model missing (with the command to fix it).

## Tuning
* **Category threshold** (`RESQTEXT_LABEL_THRESHOLD`): lower it for higher recall on rescue and medical needs, at the cost of more false alarms. Compare recall in Model Evaluation.
* **Similarity threshold**: `training/evaluate_model.py` sweeps thresholds on *validation* pairs, picks the best F1, then reports once on *test* pairs. To tune on your data, add labelled pairs to `data/duplicate_pairs.csv` (`val`/`test` splits), rerun training, and set `RESQTEXT_SIMILARITY_THRESHOLD`. Prefer higher recall here: a reviewer can dismiss a false suggestion cheaply.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "No trained classifier found" | `python -m training.train_classifier` or use the button in the app. |
| "Model artifact ... could not be loaded" | scikit-learn version changed; retrain. |
| `Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm`. The app still runs on rules + gazetteer, with weaker extraction. |
| Duplicate method says "offline fallback" | Expected unless Sentence Transformers is installed and enabled. |
| Map has no tiles | Tiles load from OpenStreetMap in your browser; needs internet. |
| Port in use | `streamlit run app.py --server.port 8502`. |
| Scripts disabled in PowerShell | `Set-ExecutionPolicy -Scope Process Bypass`. |

## Known limitations
* Bundled data is synthetic, template-based, English only. Reported scores show the pipeline works, not real-world accuracy.
* The classifier is a baseline; "General information" and "Infrastructure damage" have the lowest recall on the synthetic test set.
* The offline duplicate fallback is lexical: it misses paraphrases (see Model Evaluation). Use Sentence Transformers for semantics.
* The gazetteer covers 21 places. Unlisted places stay "unverified" unless geocoding is enabled or a reviewer corrects them.
* The duplicate scan compares against the newest 2,000 reports and recomputes embeddings; it is not optimised for very large volumes.
* No authentication or role control; do not expose it publicly. Reports can contain personal data: restrict access and retention.
* English text only; no Hindi/Marathi support yet.

## Future improvements
Multilingual models, a fine-tuned transformer, embedding cache/vector index, authentication and audit log, active learning from reviewer corrections, SMS/WhatsApp ingestion, severity-aware prioritisation validated with responders.

## Hackathon pitch (60 seconds)
During a flood, control rooms drown in messages: misspelled, repeated, half-located, several needs at once. ResQText turns that stream into a reviewable picture. It labels each need, pulls out where, how many and what, and shows the exact words that support it. It flags look-alike reports without deleting any, shows where needs are rising, and keeps every decision with a human who can correct the machine. It runs fully offline on free tools, and says plainly when it is unsure.

**Demo walkthrough (4 minutes):** (1) Settings > load demo reports. (2) NLP Analysis > "Trapped, needs water (Kalwa)", then the typo example. (3) Overview > review queue. (4) Duplicate Detection > confirm one, reject one with different locations. (5) Emerging Needs > explain the caveat. (6) Human Review > correct a label, export CSV. (7) Model Evaluation > be upfront that it is synthetic data.
