import json

import pytest

from app import extract
from app.extract import ExtractionError, extract_job
from tests.fakes import FakeClient

GOOD = json.dumps({
    "title": "Senior Python Engineer",
    "company": "Nordic Labs",
    "location": "Riga",
    "work_mode": "hybrid",
    "required_skills": ["Python", "FastAPI"],
    "nice_to_have_skills": ["Kubernetes"],
    "years_experience": 5,
    "summary": "Build FastAPI services.",
})

# "moon" is not an allowed work_mode, so validation must fail
BAD = json.dumps({"title": "Engineer", "work_mode": "moon", "summary": "x"})

JOB_TEXT = "Senior Python Engineer at Nordic Labs, Riga. Required: Python, FastAPI."


async def test_valid_json_works_first_time(monkeypatch):
    fake = FakeClient([GOOD])
    monkeypatch.setattr(extract, "client", fake)

    result = await extract_job(JOB_TEXT)

    assert result["attempts"] == 1
    assert result["job"]["company"] == "Nordic Labs"
    assert len(fake.calls) == 1


async def test_invalid_json_triggers_retry_with_error_message(monkeypatch):
    fake = FakeClient([BAD, GOOD])
    monkeypatch.setattr(extract, "client", fake)

    result = await extract_job(JOB_TEXT)

    assert result["attempts"] == 2
    # the second call must show the model what was wrong
    second_call_last_message = fake.calls[1]["messages"][-1]["content"]
    assert "Your JSON was invalid" in second_call_last_message
    # tokens from both attempts are added up (10 + 10 input, 5 + 5 output)
    assert result["input_tokens"] == 20
    assert result["output_tokens"] == 10


async def test_gives_up_after_three_attempts(monkeypatch):
    fake = FakeClient([BAD, BAD, BAD])
    monkeypatch.setattr(extract, "client", fake)

    with pytest.raises(ExtractionError):
        await extract_job(JOB_TEXT)

    assert len(fake.calls) == 3


async def test_job_text_is_wrapped_in_tags_and_schema_is_sent(monkeypatch):
    fake = FakeClient([GOOD])
    monkeypatch.setattr(extract, "client", fake)

    await extract_job(JOB_TEXT)

    call = fake.calls[0]
    assert "<job_post>" in call["messages"][1]["content"]
    assert "properties" in call["format"]      # the JSON schema was passed
    assert call["options"]["temperature"] == 0