import pytest
from fastapi.testclient import TestClient

import config
from api.main import create_app
from nlp.pipeline import Pipeline


@pytest.fixture()
def client(pipe):
    return TestClient(create_app(pipe))


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["model_loaded"] is True and "disclaimer" in j


def test_analyze_uses_real_pipeline(client):
    r = client.post("/messages/analyze", json={"text": "Five people need drinking water in Kalwa, we are trapped"})
    a = r.json()["analysis"]
    assert r.status_code == 200 and a["location"]["name"] == "Kalwa" and a["people_mentioned"] == "5"
    assert client.get("/messages").json()["total"] == 0  # analyze without save stores nothing


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 5001}, {"text": 5}])
def test_analyze_validation_errors(client, payload):
    r = client.post("/messages/analyze", json=payload)
    assert r.status_code == 422 and r.json()["detail"] == "Invalid request"


def test_save_batch_get_duplicates_review_flow(client):
    r = client.post("/messages/analyze", json={"text": "Need water in Kurla", "save": True, "timestamp": "2024-07-26T10:00:00", "is_simulated": True}).json()
    mid = r["message_id"]
    b = client.post("/messages/batch", json={"messages": [{"text": "Need water in Kurla", "timestamp": "2024-07-26T12:00:00"},
                                                          {"text": "  "}]}).json()
    assert b["inserted"] == 1 and len(b["failed"]) == 1
    assert client.get(f"/messages/{mid}").json()["is_simulated"] == 1
    assert client.get(f"/messages/{mid}/duplicates").json()["suggestions"]
    assert client.get("/messages", params={"category": "Not real"}).status_code == 422
    assert client.post("/reviews", json={"message_id": mid, "reviewer": "Asha", "status": "approved"}).status_code == 201
    assert client.get("/messages", params={"review_status": "approved"}).json()["total"] == 1
    s = client.get("/analytics/summary").json()
    assert s["counts"]["messages"] == 2 and "caveat" in s


def test_not_found_and_bad_reviews(client):
    assert client.get("/messages/MSG-NOPE").status_code == 404
    assert client.get("/messages/MSG-NOPE/duplicates").status_code == 404
    assert client.post("/reviews", json={"message_id": "MSG-NOPE", "reviewer": "A", "status": "approved"}).status_code == 404
    mid = client.post("/messages/analyze", json={"text": "Need water in Kurla", "save": True}).json()["message_id"]
    assert client.post("/reviews", json={"message_id": mid, "reviewer": "A", "status": "bogus"}).status_code == 422
    assert client.post("/duplicates/9999/decision", json={"decision": "confirmed", "reviewer": "A"}).status_code == 404
    assert client.patch(f"/messages/{mid}/location", json={"name": "Thane", "lat": 19.2}).status_code == 422


def test_evaluation_endpoint(client):
    assert client.get("/evaluation").status_code == 200


def test_missing_model_returns_503_and_app_still_starts(db, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path / "empty")
    c = TestClient(create_app(Pipeline(db)))
    assert c.get("/health").json()["model_loaded"] is False
    r = c.post("/messages/analyze", json={"text": "Need water"})
    assert r.status_code == 503 and "train_classifier" in r.json()["detail"]
    assert c.get("/evaluation").status_code == 404
