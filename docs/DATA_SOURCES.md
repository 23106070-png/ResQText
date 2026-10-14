# Real-world data sources (verified against official documentation, 13-14 Oct 2026)

| Source | What it provides | Access | Key rules | Status in this project |
|---|---|---|---|---|
| **NDMA Sachet** | Official Indian alerts (IMD weather, CWC floods, state SDMAs) as RSS + CAP 1.2 XML | Public, no key. Feed: `https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml` | Official integration guide **requires ETag caching** (If-None-Match, handle 304). Feed metadata says "public domain"; I found no separate redistribution terms, so keep attribution and confirm with NDMA before republishing. | Adapter written to the guide and tested on mocks using **real captured RSS/CAP samples**. Not yet fetched live from my environment. |
| **GDACS** | Global events: earthquakes, cyclones, floods, volcanoes, droughts, wildfires | Free, no key. `.../gdacsapi/api/events/geteventlist/SEARCH` (GeoJSON, max 100 per page, `pagenumber`) | Credit "Global Disaster Awareness and Coordination System, GDACS". Locations are event centroids. | Adapter tested on mocks using **real captured features**. Not yet fetched live from my environment. |
| **ReliefWeb API v2** | Humanitarian situation reports | **Needs a pre-approved appname** (since 1 Nov 2025). Max 1000 calls/day, 1000 entries/call. | Reports may contain copyrighted material of the original publisher: we store a short excerpt (default 3000 chars), show a shorter one, and always link back. | Adapter tested on mocks only; fixture **values are invented** (field names follow the official docs). **Cannot run without your approved appname.** |

## What you must do yourself
1. **ReliefWeb:** request an appname at https://apidoc.reliefweb.int/parameters#appname (approval is manual; use your project name). Then set `RELIEFWEB_APPNAME`.
2. **Production use:** choose strong values for `RESQTEXT_ADMIN_PASSCODE` and `RESQTEXT_API_KEY`, and put the app behind HTTPS (a reverse proxy such as Caddy/nginx or your host's TLS). Do not expose `streamlit run` or `uvicorn --reload` directly to the internet.
3. **Sachet from a cloud server:** a third-party report says NDMA's firewall can return 403 to some hosting IP ranges. If that happens the status page shows "Failing" with the reason; it works from normal home/office connections as far as I could verify.

## How each source is used
* **Official events/alerts** are stored as records of what an official body *published*. They are never classified into needs: "flood in district X" does not mean any person there needs rescue.
* **ReliefWeb reports** are classified by the existing model, but always marked **Needs human review** because the model was trained only on synthetic messages. Set `RESQTEXT_NLP_VALIDATED_ON_REAL=1` only after validating on labelled real reports.
* **Non-English text** (most Sachet headlines are Hindi, Telugu, Bengali, etc.) is stored and linked but not classified.
* **Coordinates** exist only when the source supplied them (GDACS centroid) or an alert area matches exactly one place in the small built-in gazetteer. Everything else is "not mapped".

## Labels used on screen
| Label | Meaning |
|---|---|
| Official source | Published by an official body. Not independently checked here. |
| Publisher-reported | Published by a humanitarian organisation via ReliefWeb. Not verified here. |
| Unverified | Submitted report or anything without a basis for trust. |
| Verified by reviewer | A named reviewer checked the record against its source. It does **not** mean "officially confirmed". |
| Marked incorrect / duplicate / needs investigation | Reviewer decisions. |
| NLP-inferred | Produced by the model. Unconfirmed. |

## Refresh and freshness
* Per-source minimum intervals; back-off after errors (never bypassed by the manual button); manual refresh has a 60 s floor.
* ReliefWeb call budget: 800/day by default (below the documented 1000).
* A source is "connected" only after a successful fetch is recorded. "Stale" means no success for 3 intervals.
* Updates to an item you already reviewed are **held back** and shown to you; nothing is overwritten unless you accept it.
