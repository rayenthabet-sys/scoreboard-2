"""
Gatekeeper Agent — Groq API (Llama 3)
Decides whether a submitted idea is relevant to mental health de-stigmatisation.
Also produces an innovation sub-score.
"""

import os
import json
import re
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from models import GatekeeperResult

SYSTEM_PROMPT = """Tu es le Gatekeeper de la plateforme Idéathon PACTE, un événement dédié à la santé mentale 
et à la dé-stigmatisation du recours aux professionnels de santé.

Ton rôle est d'évaluer si l'idée soumise est :
1. Pertinente par rapport à la santé mentale, au bien-être psychologique, ou à la réduction du stigma.
2. Respectueuse et non offensante.
3. Suffisamment développée pour être analysée.

Réponds UNIQUEMENT avec un JSON valide (sans markdown) de la forme :
{
  "status": "relevant" | "rejected",
  "reason": "<explication courte en français>",
  "innovation_score": <nombre entre 0 et 100>
}

Le champ innovation_score doit refléter l'originalité et la créativité de l'idée.
"""


def run_gatekeeper(idea_text: str) -> tuple[GatekeeperResult, float]:
    """
    Returns (GatekeeperResult, innovation_score).
    """
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Idée soumise :\n\n{idea_text}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()

    data = json.loads(raw)
    result = GatekeeperResult(
        status=data["status"],
        reason=data["reason"],
    )
    innovation_score = float(data.get("innovation_score", 50.0))
    return result, innovation_score
