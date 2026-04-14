from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pacte.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class IdeaRecord(Base):
    __tablename__ = "ideas"

    id = Column(Integer, primary_key=True, index=True)
    participant_name = Column(String(100), nullable=False)
    school = Column(String(50), nullable=False)
    idea_text = Column(Text, nullable=False)
    status = Column(String(20), nullable=False)  # relevant | rejected
    gatekeeper_reason = Column(Text)
    themes = Column(Text)  # JSON-encoded list
    impact_score = Column(Float, default=0.0)
    feasibility_score = Column(Float, default=0.0)
    innovation_score = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0)
    enrichment_text = Column(Text)
    similar_solutions = Column(Text)  # JSON-encoded list
    submitted_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_idea(db, result) -> IdeaRecord:
    record = IdeaRecord(
        participant_name=result.participant_name,
        school=result.school,
        idea_text=result.idea_text,
        status=result.status,
        gatekeeper_reason=result.gatekeeper_reason,
        themes=json.dumps(result.themes, ensure_ascii=False),
        impact_score=result.impact_score,
        feasibility_score=result.feasibility_score,
        innovation_score=result.innovation_score,
        final_score=result.final_score,
        enrichment_text=result.enrichment_text,
        similar_solutions=json.dumps(result.similar_solutions, ensure_ascii=False),
        submitted_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_leaderboard(db, limit: int = 20) -> list[IdeaRecord]:
    return (
        db.query(IdeaRecord)
        .filter(IdeaRecord.status == "relevant")
        .order_by(IdeaRecord.final_score.desc())
        .limit(limit)
        .all()
    )
