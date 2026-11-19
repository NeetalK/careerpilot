from fastapi.testclient import TestClient

from app import main
from app.extract import ExtractionError

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_rejects_text_that_is_too_short():
    response = client.post("/extract", json={"text": "too short"})
    assert response.status_code == 422      # FastAPI's validation error


def test_extract_returns_the_result(monkeypatch):
    async def fake_extract(text):
        return {"job": {"title": "X"}, "attempts": 1}

    monkeypatch.setattr(main, "extract_job", fake_extract)

    response = client.post("/extract", json={"text": "a" * 50})

    assert response.status_code == 200
    assert response.json()["attempts"] == 1


def test_extract_returns_502_when_the_model_keeps_failing(monkeypatch):
    async def failing_extract(text):
        raise ExtractionError("no valid JSON")

    monkeypatch.setattr(main, "extract_job", failing_extract)

    response = client.post("/extract", json={"text": "a" * 50})

    assert response.status_code == 502
    assert "no valid JSON" in response.json()["detail"]