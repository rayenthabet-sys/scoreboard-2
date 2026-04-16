# db.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Add it to your .env file: "
        "DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require"
    )


def get_connection():
    # Pass as dsn= so psycopg2 correctly parses ?sslmode=require from the URL
    return psycopg2.connect(dsn=DATABASE_URL)


@contextmanager
def get_cursor():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_tables():
    with get_cursor() as (cur, conn):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ideas (
                id                SERIAL PRIMARY KEY,
                participant_name  VARCHAR(100) NOT NULL,
                school            VARCHAR(50)  NOT NULL,
                idea_text         TEXT         NOT NULL,
                status            VARCHAR(20)  NOT NULL,
                gatekeeper_reason TEXT,
                themes            TEXT,
                impact_score      FLOAT DEFAULT 0.0,
                feasibility_score FLOAT DEFAULT 0.0,
                innovation_score  FLOAT DEFAULT 0.0,
                final_score       FLOAT DEFAULT 0.0,
                enrichment_text   TEXT,
                similar_solutions TEXT,
                submitted_at      TIMESTAMP DEFAULT NOW()
            )
        """)


def get_db():
    """FastAPI dependency — yields a dummy object, real connection is per-operation."""
    yield None


def save_idea(db, result) -> dict:
    with get_cursor() as (cur, conn):
        cur.execute("""
            INSERT INTO ideas (
                participant_name, school, idea_text, status,
                gatekeeper_reason, themes, impact_score, feasibility_score,
                innovation_score, final_score, enrichment_text,
                similar_solutions, submitted_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            result.participant_name,
            result.school,
            result.idea_text,
            result.status,
            result.gatekeeper_reason,
            json.dumps(result.themes, ensure_ascii=False),
            result.impact_score,
            result.feasibility_score,
            result.innovation_score,
            result.final_score,
            result.enrichment_text,
            json.dumps(result.similar_solutions, ensure_ascii=False),
            datetime.utcnow(),
        ))
        return cur.fetchone()


def get_leaderboard(db, limit: int = 20) -> list[dict]:
    with get_cursor() as (cur, conn):
        cur.execute("""
            SELECT
                participant_name, school, idea_text, themes,
                final_score, impact_score, feasibility_score,
                innovation_score, submitted_at
            FROM ideas
            WHERE status = 'relevant'
            ORDER BY final_score DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()

    result = []
    for row in rows:
        row = dict(row)
        row["themes"] = json.loads(row["themes"] or "[]")
        result.append(row)
    return result