from typing import Literal
from pydantic import BaseModel, Field


class JobPost(BaseModel):
    title: str = Field(description="Job title exactly as written in the post")
    company: str | None = Field(default=None, description="Company name, or null if not written")
    location: str | None = Field(default=None, description="City or country of the job, or null if not written")
    work_mode: Literal["remote", "hybrid", "onsite", "unknown"] = Field(
        default="unknown",
        description="remote, hybrid or onsite. Use unknown only if the post says nothing about it",
    )
    required_skills: list[str] = Field(default=[], description="Skills the post says are required")
    nice_to_have_skills: list[str] = Field(default=[], description="Skills the post says are a bonus")
    years_experience: int | None = Field(default=None, description="Minimum years of experience, or null")
    summary: str = Field(description="One or two full sentences about what the person will do in this job")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20, max_length=200_000)
    source: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)