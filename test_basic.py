import pytest
from pydantic import ValidationError

from app.schemas import AskRequest
from app.services.ingest import build_chunks, load_schemes
from app.services.prompt import build_messages

REQUIRED_FIELDS = [
    "id", "name", "category", "summary", "eligibility",
    "benefits", "documents_required", "how_to_apply", "source_url",
]


def test_every_scheme_has_required_fields():
    for s in load_schemes():
        for field in REQUIRED_FIELDS:
            assert s.get(field), f"{s.get('name')} is missing '{field}'"
        assert s["source_url"].startswith("https://")


def test_chunks_are_built_with_unique_ids():
    schemes = load_schemes()
    chunks = build_chunks(schemes)
    assert len(chunks) == len(schemes) * 4
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_prompt_contains_context_and_question():
    chunk = {"scheme": "Test Scheme", "text": "Some text.", "source_url": "https://x.gov.in"}
    messages = build_messages("Am I eligible for anything?", [chunk])
    assert messages[0]["role"] == "system"
    assert "Test Scheme" in messages[1]["content"]
    assert "Am I eligible for anything?" in messages[1]["content"]


def test_short_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")
