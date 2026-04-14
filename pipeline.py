"""
Pipeline — LangChain Orchestration
Wires: Gatekeeper (Groq) → Analyser (Gemini) → Validator (Tavily) → Scorer
"""

from models import IdeaSubmission, PipelineResult
from gatekeeper import run_gatekeeper
from analyser import run_analyser
from validator import run_validator
from scorer import compute_score
from datetime import datetime


def run_pipeline(submission: IdeaSubmission) -> PipelineResult:
    """
    Execute the full 3-agent pipeline for a submitted idea.

    Steps:
    1. Gatekeeper (Groq/Llama3)  — relevance check + innovation score
    2. Analyser  (Gemini Flash)   — theme extraction + impact score
    3. Validator (Tavily Search)  — similar solutions + feasibility score
    4. Scorer    (pure function)  — weighted final score
    """

    # ── Step 1: Gatekeeper ──────────────────────────────────────────────────
    gatekeeper_result, innovation_score = run_gatekeeper(submission.idea_text)

    if gatekeeper_result.status == "rejected":
        return PipelineResult(
            idea_text=submission.idea_text,
            participant_name=submission.participant_name,
            school=submission.school,
            status="rejected",
            gatekeeper_reason=gatekeeper_result.reason,
            themes=[],
            impact_score=0.0,
            feasibility_score=0.0,
            innovation_score=0.0,
            final_score=0.0,
            enrichment_text="",
            similar_solutions=[],
            submitted_at=datetime.utcnow(),
        )

    # ── Step 2: Analyser ────────────────────────────────────────────────────
    analyser_result = run_analyser(submission.idea_text)

    # ── Step 3: Validator ───────────────────────────────────────────────────
    validator_result = run_validator(submission.idea_text, analyser_result.themes)

    # ── Step 4: Scorer ──────────────────────────────────────────────────────
    final_score = compute_score(
        impact_score=analyser_result.impact_score,
        feasibility_score=validator_result.feasibility_score,
        innovation_score=innovation_score,
    )

    return PipelineResult(
        idea_text=submission.idea_text,
        participant_name=submission.participant_name,
        school=submission.school,
        status="relevant",
        gatekeeper_reason=gatekeeper_result.reason,
        themes=analyser_result.themes,
        impact_score=analyser_result.impact_score,
        feasibility_score=validator_result.feasibility_score,
        innovation_score=innovation_score,
        final_score=final_score,
        enrichment_text=validator_result.enrichment_text,
        similar_solutions=validator_result.similar_solutions,
        submitted_at=datetime.utcnow(),
    )
