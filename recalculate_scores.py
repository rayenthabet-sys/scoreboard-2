import json
import time
import os
from db import get_cursor
from models import IdeaSubmission
from pipeline import run_pipeline

PROGRESS_FILE = "recalculated_ids.txt"

def get_processed_ids():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(idea_id):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{idea_id}\n")

def recalculate_all():
    processed_ids = get_processed_ids()
    print(f"Fetching ideas from the database... ({len(processed_ids)} already processed)")
    
    # Get all submitted ideas that are currently relevant
    with get_cursor() as (cur, conn):
        cur.execute("SELECT id, participant_name, school, idea_text FROM ideas WHERE status = 'relevant'")
        rows = cur.fetchall()
        
    to_process = [row for row in rows if str(row['id']) not in processed_ids]
    print(f"Found {len(to_process)} ideas to recalculate (Skipped {len(rows) - len(to_process)}).")
    
    for row in to_process:
        idea_id = row['id']
        print(f"\nProcessing ID {idea_id}: {row['participant_name'][:20]}...")
        
        submission = IdeaSubmission(
            participant_name=row['participant_name'],
            school=row['school'],
            idea_text=row['idea_text']
        )
        
        try:
            # Re-run the evaluation pipeline to get new scores
            result = run_pipeline(submission)
            
            # Update the database with the newly generated results
            with get_cursor() as (cur, conn):
                cur.execute("""
                    UPDATE ideas SET 
                        status = %s,
                        gatekeeper_reason = %s,
                        themes = %s,
                        impact_score = %s,
                        feasibility_score = %s,
                        innovation_score = %s,
                        final_score = %s,
                        enrichment_text = %s,
                        similar_solutions = %s
                    WHERE id = %s
                """, (
                    result.status,
                    result.gatekeeper_reason,
                    json.dumps(result.themes, ensure_ascii=False),
                    result.impact_score,
                    result.feasibility_score,
                    result.innovation_score,
                    result.final_score,
                    result.enrichment_text,
                    json.dumps(result.similar_solutions, ensure_ascii=False),
                    idea_id
                ))
            
            mark_as_processed(idea_id)
            print(f"  -> Done! New Final Score: {result.final_score} (Impact: {result.impact_score}, Innovation: {result.innovation_score})")
            
            # Small delay to respect rate limits (Tokens Per Minute)
            time.sleep(3) 
            
        except Exception as e:
            error_msg = str(e)
            print(f"  -> Error processing ID {idea_id}: {error_msg}")
            if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                print("🛑 Rate limit reached. Stopping for now. Run the script again later to continue.")
                break
            
    print("\nRecalculation batch finished!")

if __name__ == "__main__":
    recalculate_all()
