"""
Analyser Agent — Gemini 1.5 Flash
Extracts thematic tags and computes the impact sub-score.
"""

import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from models import AnalyserResult

SYSTEM_PROMPT = """Tu es l'analyste thématique de la plateforme Idéathon PACTE.
Ton rôle est d'extraire les thématiques clés d'une idée liée à la santé mentale 
et d'évaluer son impact social potentiel sur la réduction du stigma.

Réponds UNIQUEMENT avec un JSON valide (sans markdown) de la forme :
{
  "themes": ["thème1", "thème2", "thème3"],
  "impact_score": <nombre entre 0 et 100>,
  "impact_justification": "<explication courte en français>"
}

Thèmes possibles (non exhaustifs) : 
Sensibilisation, Accessibilité, Prévention, Inclusion, Éducation, Innovation technologique, 
Communauté, Bien-être, Résilience, Soutien par les pairs, Thérapie, Stigmatisation, etc.

L'impact_score doit refléter :
- La portée potentielle de l'idée (combien de personnes peuvent en bénéficier)
- La profondeur de l'impact sur la réduction du stigma
- Le réalisme de l'impact positif attendu
"""


def run_analyser(idea_text: str) -> AnalyserResult:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Idée à analyser :\n\n{idea_text}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    data = json.loads(raw)
    return AnalyserResult(
        themes=data["themes"],
        impact_score=float(data["impact_score"]),
    )
