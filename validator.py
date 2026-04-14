"""
Validator Agent — Tavily Search (RAG)
Checks for similar existing solutions and enriches the idea with external context.
Also computes the feasibility sub-score based on found evidence.
"""

import os
from tavily import TavilyClient
from models import ValidatorResult


def run_validator(idea_text: str, themes: list[str]) -> ValidatorResult:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Build a focused search query from the idea and its themes
    theme_str = ", ".join(themes[:3]) if themes else "santé mentale"
    query = f"solutions existantes santé mentale stigma {theme_str}: {idea_text[:200]}"

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    # Extract similar solutions titles + URLs
    similar_solutions = []
    for result in response.get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        if title:
            similar_solutions.append(f"{title} — {url}")

    # Use Tavily's synthesised answer as enrichment context
    enrichment_text = response.get("answer") or "Aucun enrichissement disponible."

    # Feasibility score: more existing similar solutions → higher feasibility evidence
    num_results = len(similar_solutions)
    if num_results == 0:
        feasibility_score = 40.0  # Innovative but unproven
    elif num_results <= 2:
        feasibility_score = 65.0
    elif num_results <= 4:
        feasibility_score = 80.0
    else:
        feasibility_score = 90.0

    return ValidatorResult(
        similar_solutions=similar_solutions,
        enrichment_text=enrichment_text,
        feasibility_score=feasibility_score,
    )
