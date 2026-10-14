import pytest

from nlp.pipeline import CSVValidationError, parse_csv


def test_insert_and_retrieve(pipe):
    r = pipe.ingest("Five people need drinking water in Kalwa", "2024-07-26T10:00:00", is_simulated=True)
    m = pipe.db.get_message(r["message_id"])
    assert m["original_text"] == "Five people need drinking water in Kalwa"
    assert m["normalized_text"] != m["original_text"] and m["location_text"] == "Kalwa" and m["is_simulated"] == 1
    assert m["predicted_labels"] and m["entities"] and m["review_status"] == "pending"


def test_reimport_same_report_skipped(pipe):
    a = pipe.ingest("Need water in Kalwa", "2024-07-26T10:00:00")
    b = pipe.ingest("Need water in Kalwa", "2024-07-26T10:00:00")
    assert a["created"] and not b["created"] and a["message_id"] == b["message_id"]


def test_identical_text_different_timestamps_kept(pipe):
    a = pipe.ingest("Need water in Kalwa", "2024-07-26T10:00:00")
    b = pipe.ingest("Need water in Kalwa", "2024-07-26T18:00:00")
    assert b["created"] and a["message_id"] != b["message_id"]
    assert b["duplicates"] and b["duplicates"][0]["other_id"] == a["message_id"]  # flagged, not deleted
    assert pipe.db.counts()["messages"] == 2


def test_sql_injection_text_is_stored_literally(pipe):
    evil = "Need water'); DROP TABLE messages;-- in Kalwa"
    r = pipe.ingest(evil, "2024-07-26T10:00:00")
    assert pipe.db.get_message(r["message_id"])["original_text"] == evil and pipe.db.counts()["messages"] == 1


def test_empty_text_rejected(db):
    with pytest.raises(ValueError):
        db.add_message("   ")


def test_review_roundtrip_and_validation(pipe):
    mid = pipe.ingest("Need water in Kalwa", "2024-07-26T10:00:00")["message_id"]
    pipe.db.add_review(mid, "Asha", "corrected", ["Medical assistance"], "Thane", 19.2183, 72.9781, "checked by phone")
    m = pipe.db.get_message(mid)
    assert m["final_labels"] == ["Medical assistance"] and m["location_status"] == "manual" and m["review_status"] == "corrected"
    assert len(pipe.db.reviewed_export()) == 1
    with pytest.raises(ValueError):
        pipe.db.add_review(mid, "", "approved")
    with pytest.raises(ValueError):
        pipe.db.add_review(mid, "Asha", "corrected", ["Not a label"])
    with pytest.raises(ValueError):
        pipe.db.add_review(mid, "Asha", "corrected", None, "X", 95.0, 10.0)
    with pytest.raises(KeyError):
        pipe.db.add_review("MSG-NOPE", "Asha", "approved")


def test_duplicate_decision(pipe):
    pipe.ingest("Need water in Kalwa", "2024-07-26T10:00:00")
    pipe.ingest("Need water in Kalwa", "2024-07-26T11:00:00")
    sid = pipe.db.list_duplicates(status="pending")[0]["id"]
    pipe.db.decide_duplicate(sid, "rejected", "Asha")
    assert pipe.db.list_duplicates(status="rejected")


def test_csv_invalid_empty_missing_columns(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"")
    with pytest.raises(CSVValidationError):
        parse_csv(f)
    f.write_text("message,time\nhello,now\n")
    with pytest.raises(CSVValidationError, match="text"):
        parse_csv(f)
    f.write_bytes(b"\x00\xff\xfe binary \x00")
    with pytest.raises(CSVValidationError):
        parse_csv(f)
    f.write_text("text\n  \n\n")
    with pytest.raises(CSVValidationError):
        parse_csv(f)


def test_csv_warnings_and_batch(tmp_path, pipe):
    f = tmp_path / "ok.csv"
    f.write_text("text,timestamp\nNeed water in Kurla,2024-07-26T10:00:00\n,2024-07-26T10:05:00\nNeed food in Sion,not-a-date\n")
    df, warnings = parse_csv(f)
    assert len(df) == 2 and len(warnings) == 2
    s = pipe.ingest_dataframe(df)
    assert s["inserted"] == 2 and not s["failed"]
    assert pipe.ingest_dataframe(df)["skipped_existing"] >= 1
