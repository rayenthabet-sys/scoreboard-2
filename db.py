# db.py
import os
import json
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()  # ← must be here too, db.py is imported before main.py runs

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# Connection pool — Neon free tier allows max 10 connections
_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=5,
    dsn=DATABASE_URL,
    connect_timeout=10,          # ← stops the infinite hang
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=5,
)


@contextmanager
def get_cursor():
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)       # ← return to pool, don't close


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