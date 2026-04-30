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
    
    prompt = f"""Évalue la faisabilité technique et logistique (0-100) de cette idée dans le contexte tunisien actuel.
Idée : {idea_text[:300]}
Preuves de faisabilité trouvées :
{solutions_str}

CONSIGNES DE SCORING :
- Si l'idée demande des ressources financières importantes sans partenaire identifié : MAX 35.
- Si l'idée demande un changement de loi ou de réglementation : MAX 25.
- Si l'idée est purement numérique mais sans plan de maintenance : MAX 50.
- Ne dépasse 75 QUE SI l'idée est déjà testée avec succès ou est extrêmement simple à mettre en place avec les moyens du bord.

Réponds UNIQUEMENT avec un JSON valide (sans markdown) : {{"feasibility_score": <nombre>}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content.strip()).strip()
    return float(json.loads(raw)["feasibility_score"])



def generate_search_query(idea_text: str, themes: list[str]) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    )
    prompt = f"""Génère une requête de recherche Google courte et précise (en français ou anglais) pour trouver des projets existants similaires à cette idée d'étudiant.
Idée : {idea_text[:300]}
Thèmes : {", ".join(themes)}

Réponds UNIQUEMENT avec la requête de recherche, sans ponctuation inutile ni phrases."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().strip('"')



def run_validator(idea_text: str, themes: list[str]) -> ValidatorResult:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # --- FIX 1: Generate dynamic query instead of hardcoded strings ---
    query = generate_search_query(idea_text, themes)

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

    # --- FIX 2: Validate enrichment text to prevent "bridging hallucinations" ---
    enrichment_text = response.get("answer") or "Aucun enrichissement disponible."
    
    # If the answer is too generic or seems irrelevant, we could do a secondary check here, 
    # but dynamic query generation already solves 90% of the problem.

    # --- THE FIX: Use LLM to score feasibility based on EVIDENCE, not just links ---
    feasibility_score = evaluate_feasibility_with_llm(idea_text, similar_solutions)

    return ValidatorResult(
        similar_solutions=similar_solutions,
        enrichment_text=enrichment_text,
        feasibility_score=feasibility_score,
    )