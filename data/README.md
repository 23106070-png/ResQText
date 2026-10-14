# Data

## Files
| File | What it is |
|---|---|
| `sample_messages.csv` | **SYNTHETIC** development dataset (template-generated, 1,380 rows). Not real reports. |
| `duplicate_pairs.csv` | **SYNTHETIC** labelled message pairs for duplicate-detection evaluation. |
| `gazetteer.csv` | 21 demo place names (Thane / Mumbai region) with **approximate area-centre** coordinates for offline mapping. Verify before operational use. |

Regenerate the synthetic files with `python -m training.synthetic_data` (seeded, reproducible).

## Classification CSV schema (`sample_messages.csv`)
| Column | Required | Description |
|---|---|---|
| `text` | yes | Original message text |
| `labels` | yes | One or more categories separated by `\|`, e.g. `Rescue assistance\|Food and drinking water`. Aliases are normalised (see `training/prepare_dataset.py`). |
| `message_id` | no | Your own identifier |
| `timestamp` | no | ISO 8601, e.g. `2024-07-26T10:15:00` |
| `source` | no | e.g. `sms`, `twitter`, `helpline` |
| `is_simulated` | no | `1` for synthetic/demo rows |
| `split` | no | `train` / `val` / `test`. If present it is respected; otherwise a stratified split is made. |

Categories: Rescue assistance, Medical assistance, Food and drinking water, Shelter and evacuation, Infrastructure damage, General information.

## Upload CSV schema (dashboard / `POST /messages/batch`)
`text` (required); optional `timestamp`, `source`, `location`, `message_id`, `is_simulated`.

## Duplicate pairs schema (`duplicate_pairs.csv`)
`text_a, text_b, location_a, location_b, is_duplicate (0/1), pair_type, split`.

## Leakage protection
Templates are split into train/val/test pools **before** messages are generated, so a test sentence pattern never appears in training. Normalised duplicate texts are dropped before splitting and `assert_no_leakage()` fails if any text appears in two splits.
Even so, the data is far simpler than real messages: scores on it are **not** real-world performance.

## Real humanitarian datasets (checked 12 Oct 2026)
Nothing here is downloaded automatically. Check each source's current terms before use.

| Dataset | Source | Availability notes |
|---|---|---|
| **HumAID** (~77K human-labelled tweets, 19 disasters 2016-2019, English) | https://crisisnlp.qcri.org/humaid_dataset | Publicly listed by QCRI. Set 1 (about 47.9K tweets) has a download link; set 2 (about 29.3K) must be requested through a form; the combined/event-type files hold tweet IDs and labels and need text taken from the event files. A Hugging Face loader exists (`QCRI/HumAID-all`). **I could not confirm the data licence from the pages I could read** (the Apache header on the loader covers the loader code, not the data). Confirm terms before any use beyond research. |
| **CrisisNLP resources** (Nepal earthquake, Queensland floods, Hurricane Sandy, Joplin tornado, multimodal sets) | https://crisisnlp.qcri.org | Several corpora; some list tweet IDs only, which require re-fetching tweet text through the platform API. Licences differ per resource. |

HumAID is **tweets and single-label**; ResQText is **multi-label** and aimed at direct reports. Mapping is approximate (`LABEL_ALIASES`), so expect a domain gap.

Using HumAID after you download it manually:
```powershell
# files unpacked under data\humaid\  (tsv files with tweet_text and class_label columns)
python -c "from training.prepare_dataset import load_humaid; load_humaid('data/humaid').to_csv('data/humaid_combined.csv', index=False)"
python -m training.train_classifier data/humaid_combined.csv
```
Rows whose HumAID label has no mapping (e.g. `requests_or_urgent_needs`) are dropped.
