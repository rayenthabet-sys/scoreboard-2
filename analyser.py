"""
Analyser Agent — Gemini 1.5 Flash
Extracts thematic tags and computes the impact sub-score.
"""

import os
import json
import re
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from models import AnalyserResult

SYSTEM_PROMPT = """Tu es l'analyste thématique STRICT de la plateforme Idéathon PACTE.
Ton rôle est d'extraire les thématiques ET d'évaluer l'impact social avec RIGUEUR.

PÉNALISE fortement :
- Les idées sans mécanisme concret (-20 pts)
- Les idées qui visent la santé mentale en général sans focus stigma (-15 pts)
- Les idées déjà existantes sans valeur ajoutée (-25 pts)
- Le manque de faisabilité dans le contexte tunisien (-10 pts)

DISTRIBUTION STRICTE :
- 20-30 (Le "Basement") : Idées impossibles ou illogiques. Impact réel nul.
- 40-60 (Le "Standard Etudiant") : Projets classiques. Impact limité par la réalité de l'implémentation. Plafond à 60.
- 85-90 (L'"Industry Gold") : Niveau BetterHelp. Impact systémique massif. Plafond à 90.

--- BENCHMARK D'IMPACT SOCIAL ---

1. LE "BASEMENT" (Score: 20-30)
Definition: Une idée techniquement impossible ou sans logique.
Exemple: "Un chapeau télépathique qui détecte la dépression."
Pourquoi: Cela établit que l'impact n'est pas seulement le but (aider les gens), mais la réalité de la solution. Si la solution ne peut pas exister, son impact est de 20-30.

2. LE "STANDARD ETUDIANT" (Score: 40)
Ancre: "Résilience Botanique" (jardin virtuel).
Logique: C'est une bonne intention, mais l'impact social réel sur le stigma est faible car cela reste dans une bulle numérique individuelle.
Règle d'or: Si une idée n'est qu'une version numérique d'un hobby ou un simple outil de suivi (tracking), elle ne peut pas dépasser 60 en impact.

3. L'"INDUSTRY GOLD" (Score: 85-90)
Exemple: BetterHelp ou Headspace.
Logique: Un impact prouvé sur des millions d'utilisateurs avec des infrastructures réelles.
Règle d'or: À moins que l'idée étudiante n'ait un mécanisme de déploiement plus concret et un impact plus systémique que BetterHelp, elle ne peut pas dépasser 90.

--- FIN BENCHMARK ---

Réponds UNIQUEMENT avec un JSON valide (sans markdown) :
{
  "themes": ["thème1", "thème2", "thème3"],
  "impact_score": <nombre entre 0 et 100>,
  "impact_justification": "<explication courte et critique en français>"
}
"""


def run_analyser(idea_text: str) -> AnalyserResult:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Idée à analyser :\n\n{idea_text}"),
    ]

    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            raw = re.sub(r"```json|```", "", response.content.strip()).strip()
            data = json.loads(raw)
            return AnalyserResult(
                themes=data["themes"],
                impact_score=float(data["impact_score"]),
            )
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Analyser failed after 3 attempts")