import pytest

import config
from nlp.classifier import DisasterClassifier, ModelNotFoundError
from nlp.entity_extraction import extract
from nlp.preprocessing import normalize_text, redact_pii

KALWA = "We are trapped due to flooding near the municipal school in Kalwa. Five people need drinking water."


def test_normalize_empty_and_none():
    assert normalize_text("") == "" and normalize_text(None) == ""


def test_typo_repair_keeps_place_names():
    out = normalize_text("Need watr and medicin in Kalwa")
    assert "water" in out and "medicine" in out and "kalwa" in out


def test_pii_redaction():
    assert "9876543210" not in redact_pii("call me on 9876543210 or a@b.com")


def test_missing_model_artifacts(tmp_path):
    with pytest.raises(ModelNotFoundError):
        DisasterClassifier.load(tmp_path)


def test_corrupt_model_artifact(tmp_path):
    (tmp_path / "classifier.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelNotFoundError):
        DisasterClassifier.load(tmp_path)


def test_empty_message_rejected(pipe):
    for bad in ["", "   ", "\n"]:
        with pytest.raises(ValueError):
            pipe.analyze(bad)


def test_multiple_needs_in_one_message(pipe):
    r = pipe.analyze(KALWA)
    assert "Rescue assistance" in r["labels"] and "Food and drinking water" in r["labels"]
    assert set(r["scores"]) == set(config.LABELS)


def test_spelling_mistakes_still_classified(pipe):
    r = pipe.analyze("pregnent woman in labour, no ambulence can reach us, need a doctr")
    assert "Medical assistance" in r["labels"]


def test_kalwa_example_structure():
    e = extract(KALWA)
    assert e.primary_location == "Kalwa" and e.people_count == "5" and "Drinking water" in e.resources
    ppl = [x for x in e.entities if x.type == "PEOPLE"][0]
    assert KALWA[ppl.start:ppl.end] == ppl.evidence  # evidence is a real span of the original text


def test_unknown_location_not_invented(pipe):
    r = pipe.analyze("my father needs insulin urgently and we have no electricity")
    assert r["location"]["name"] == "Unknown" and r["location"]["lat"] is None and r["needs_verification"]
    assert r["people_mentioned"] == "Unknown"


def test_unrecognised_place_has_no_coordinates(pipe):
    r = pipe.analyze("Trapped in Zzyzxville, please send a boat")
    assert r["location"]["status"] == "unverified" and r["location"]["lat"] is None


def test_ambiguous_place_has_no_coordinates(pipe):
    r = pipe.analyze("Need food at Shivaji Nagar")
    assert r["location"]["status"] == "ambiguous" and r["location"]["lon"] is None


def test_known_place_marked_approximate(pipe):
    loc = pipe.analyze("Water has entered houses in Kurla")["location"]
    assert loc["status"] == "verified" and "Approximate" in loc["note"]


def test_people_count_words_and_multiple_mentions():
    e = extract("Family of four and 12 elderly residents stuck")
    assert e.people_count == "12" and "Needs verification" in e.people_note
