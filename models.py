from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class IdeaSubmission(BaseModel):
    idea_text: str = Field(..., min_length=20, max_length=2000)
    participant_name: str = Field(..., min_length=2, max_length=100)
    school: str = Field(..., pattern="^(IPEST|SUP'COM|ISSHT)$")


class GatekeeperResult(BaseModel):
    status: str  # "relevant" | "rejected"
    reason: str


class AnalyserResult(BaseModel):
    themes: list[str]
    impact_score: float  # 0-100


class ValidatorResult(BaseModel):
    similar_solutions: list[str]
    enrichment_text: str
    feasibility_score: float  # 0-100


class PipelineResult(BaseModel):
    idea_text: str
    participant_name: str
    school: str
    status: str  # "relevant" | "rejected"
    gatekeeper_reason: str
    themes: list[str]
    impact_score: float
    feasibility_score: float
    innovation_score: float
    final_score: float
    enrichment_text: str
    similar_solutions: list[str]
    submitted_at: Optional[datetime] = None


class LeaderboardEntry(BaseModel):
    rank: int
    participant_name: str
    school: str
    idea_text: str
    themes: list[str]
    final_score: float
    impact_score: float
    feasibility_score: float
    innovation_score: float
    submitted_at: datetime
