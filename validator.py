"""
Validator Agent — Tavily Search (RAG)
Checks for similar existing solutions and enriches the idea with external context.
Also computes the feasibility sub-score based on found evidence.
"""


import os
from tavily import TavilyClient
from models import ValidatorResult
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Add this helper function
def evaluate_feasibility_with_llm(idea_text: str, similar_solutions: list[str]) -> float:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0, 
    )
    
    solutions_str = "\n".join(f"- {s}" for s in similar_solutions[:3]) if similar_solutions else "Aucune solution similaire trouvée."
    
    prompt = f"""Évalue la faisabilité technique et logistique (0-100) de cette idée dans le contexte tunisien.
Idée : {idea_text[:300]}
Preuves de faisabilité trouvées :
{solutions_str}
Réponds UNIQUEMENT avec un JSON valide (sans markdown) : {{"feasibility_score": <nombre>}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content.strip()).strip()
    return float(json.loads(raw)["feasibility_score"])



def run_validator(idea_text: str, themes: list[str]) -> ValidatorResult:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Make the search query stable (don't rely on fluctuating themes)
    query = f"projet étudiant santé mentale stigma université tunisie: {idea_text[:200]}"

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    similar_solutions = []
    for result in response.get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        if title:
            similar_solutions.append(f"{title} — {url}")

    enrichment_text = response.get("answer") or "Aucun enrichissement disponible."

    # --- THE FIX: Use LLM to score feasibility instead of counting links ---
    feasibility_score = evaluate_feasibility_with_llm(idea_text, similar_solutions)

    return ValidatorResult(
        similar_solutions=similar_solutions,
        enrichment_text=enrichment_text,
        feasibility_score=feasibility_score,
    )