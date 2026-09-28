"""Pydantic models: define and validate what goes into and out of the API."""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=600)
    category: Optional[str] = None  # optional filter, e.g. "Scholarship"

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        v = " ".join(v.split())  # collapse extra spaces/newlines
        if len(v) < 10:
            raise ValueError("Please describe your situation in a little more detail.")
        return v


class Source(BaseModel):
    scheme: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    mode: str  # "rag", "retrieval_only" or "no_match"
