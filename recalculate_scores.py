import json
from db import get_cursor
from models import IdeaSubmission
from pipeline import run_pipeline

def recalculate_all():
    print("Fetching ideas from the database...")
    # Get all submitted ideas that are currently relevant
    with get_cursor() as (cur, conn):
        cur.execute("SELECT id, participant_name, school, idea_text FROM ideas WHERE status = 'relevant'")
        rows = cur.fetchall()
        
    print(f"Found {len(rows)} ideas to recalculate. Starting the re-evaluation pipeline (this may take a few minutes)...")
    
    for row in rows:
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
            print(f"  -> Done! New Final Score: {result.final_score} (Impact: {result.impact_score}, Innovation: {result.innovation_score})")
            
        except Exception as e:
            print(f"  -> Error processing ID {idea_id}: {e}")
            
    print("\nRecalculation complete!")

if __name__ == "__main__":
    recalculate_all()
