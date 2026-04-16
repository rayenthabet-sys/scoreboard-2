"""
Gatekeeper Agent — Groq API (Llama 3)
Decides whether a submitted idea is relevant to mental health de-stigmatisation.
Also produces an innovation sub-score.
"""

import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from models import GatekeeperResult

SYSTEM_PROMPT = """Tu es le Gatekeeper strict de la plateforme Idéathon PACTE.

Tu dois REJETER toute idée qui :
- Est trop vague ou incomplète (moins de 2 phrases concrètes)
- N'explique pas COMMENT elle fonctionne, seulement QUOI elle fait
- Est une simple reformulation d'une solution évidente
- Manque de lien direct avec la santé mentale ou la dé-stigmatisation

Une idée DOIT contenir :
✓ Un problème clairement identifié
✓ Une proposition concrète de solution
✓ Un public cible défini
✓ Un mécanisme d'action concret

Pour l'innovation_score, utilise CES EXEMPLES comme référence fixe :

--- BENCHMARK ---

SCORE 10 — À REJETER :
Idée : "Faire une application de méditation."
Pourquoi : Une seule phrase, aucun mécanisme, aucun lien avec le stigma,
des milliers d'apps identiques existent (Calm, Headspace). Idée non développée.

SCORE 10 — À REJETER :
Idée : "Aider les étudiants autistes."
Pourquoi : 4 mots. Aucun problème défini, aucune solution proposée,
aucun mécanisme. C'est un souhait, pas une idée.

SCORE 30 — LIMITE, À REJETER sauf si très bien justifié :
Idée : "Une plateforme de consultation psychologique en ligne gratuite pour étudiants."
Pourquoi : Le concept existe déjà massivement (BetterHelp, Wisal, etc.).
La gratuité ne suffit pas comme différenciation. Aucun élément anti-stigma.

SCORE 55 — PERTINENT, innovation faible :
Idée : "Un système de parrainage entre étudiants de première année et 
étudiants seniors formés à l'écoute active, pour briser l'isolement 
et normaliser les conversations sur la santé mentale dès l'arrivée à l'université."
Pourquoi : Mécanisme clair (parrainage), public défini (L1), lien direct 
avec la normalisation du sujet. Mais le concept de mentoring existe déjà.

SCORE 80 — FORT :
Idée : "Un réseau d'ambassadeurs santé mentale certifiés dans chaque 
grande école tunisienne (IPEST, SUP'COM, ISSHT...), formés à détecter 
les signaux faibles et orienter vers des professionnels — avec un tableau 
de bord anonymisé permettant aux administrations de mesurer le bien-être 
du campus sans identifier les individus."
Pourquoi : Mécanisme systémique, ancré dans la réalité locale tunisienne,
adresse le stigma par la normalisation institutionnelle, mesurable.

--- FIN BENCHMARK ---

Réponds UNIQUEMENT avec un JSON valide (sans markdown) :
{
  "status": "relevant" | "rejected",
  "reason": "<explication courte et directe en français>",
  "innovation_score": <nombre entre 0 et 100>
}
"""


def run_gatekeeper(idea_text: str) -> tuple[GatekeeperResult, float]:
    """
    Returns (GatekeeperResult, innovation_score).
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
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
