"""
PACTE Idéathon — FastAPI Entry Point
Endpoints:
  POST /submit-idea    → run the AI pipeline, persist result
  GET  /leaderboard    → return ranked ideas (relevant only)
  GET  /               → serve index.html
  GET  /leaderboard-ui → serve leaderboard.html
"""
import collections
import collections.abc
# Patch removed aliases for Python 3.10+ compatibility
for _name in ("MutableSet", "MutableMapping", "MutableSequence", "Callable",
              "Mapping", "Sequence", "Set", "MutableMapping"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))





from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
from datetime import datetime

from models import IdeaSubmission, PipelineResult, LeaderboardEntry
from pipeline import run_pipeline
from db import create_tables, get_db, save_idea, get_leaderboard

app = FastAPI(
    title="PACTE Idéathon API",
    description="Plateforme IA de soumission et scoring d'idées — Santé Mentale SUP'COM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    create_tables()


# ── HTML Pages ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/leaderboard-ui", response_class=HTMLResponse)
def serve_leaderboard_ui():
    html_path = Path(__file__).parent / "leaderboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="leaderboard.html not found")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/submit-idea", response_model=PipelineResult)
def submit_idea(submission: IdeaSubmission, db: Any = Depends(get_db)):
    """
    Run the full AI pipeline on a submitted idea and persist the result.
    Returns the full PipelineResult (including rejection details if filtered out).
    """
    try:
        result = run_pipeline(submission)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    save_idea(db, result)
    return result


@app.get("/leaderboard", response_model=list[LeaderboardEntry])
def leaderboard(limit: int = 20, db: Any = Depends(get_db)):
    """
    Return the top N relevant ideas ranked by final score.
    """
    records = get_leaderboard(db, limit=limit)
    entries = []
    for rank, record in enumerate(records, start=1):
        entries.append(LeaderboardEntry(
            rank=rank,
            participant_name=record["participant_name"],
            school=record["school"],
            idea_text=record["idea_text"],
            themes=record["themes"],          # already decoded by get_leaderboard()
            final_score=record["final_score"],
            impact_score=record["impact_score"],
            feasibility_score=record["feasibility_score"],
            innovation_score=record["innovation_score"],
            submitted_at=record["submitted_at"],
        ))
    return entries


@app.get("/health")
def health():
    return {"status": "ok", "service": "PACTE Idéathon API"}
