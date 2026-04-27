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

Utilise CES EXEMPLES comme étalon fixe pour calibrer ton impact_score ET RESPECTE la distribution attendue :

DISTRIBUTION OBLIGATOIRE : ~50% des idées acceptées se situent entre 20–55, ~35% entre 55–72, ~12% entre 72–84, et MOINS DE 3% au-dessus de 85. Un impact de 85+ signifie un changement culturel démontrable et systémique — c'est extrêmement rare. Si tu dépasses 75, justifie-le explicitement.

--- BENCHMARK ---

IMPACT 15 :
Idée : "Créer une smartwatch qui détecte le stress en temps réel."
Pourquoi : L'impact sur la réduction du stigma est quasi nul. Mesurer le stress
ne change pas la perception sociale de la maladie mentale. Portée très limitée,
coût d'accès prohibitif pour le contexte tunisien.

IMPACT 35 :
Idée : "Une application mobile gratuite qui connecte les étudiants à des 
psychologues bénévoles avec un système de matching."
Pourquoi : L'accessibilité est réelle mais l'impact anti-stigma est indirect.
Le matching psychologue/patient existe déjà. Pas de mécanisme de changement 
culturel ou communautaire.

IMPACT 52 :
Idée : "Des capsules vidéo courtes (format Reels/TikTok) co-créées avec 
des étudiants de SUP'COM et des psychologues, montrant des témoignages 
anonymes de pairs qui ont consulté — diffusées sur les réseaux intra-campus."
Pourquoi : Mécanisme de déstigmatisation par identification (les pairs).
Portée réelle sur les réseaux. Mais impact limité au campus, pas de 
mécanisme de suivi, d'évaluation ou de changement durable. Le format
TikTok/Reels de sensibilisation est très répandu et son efficacité anti-stigma
à long terme est discutable.

IMPACT 66 :
Idée : "Un programme de certification 'Espace Safe' pour les cafés et 
espaces étudiants tunisiens : les gérants sont formés à l'écoute de 
premier niveau et affichent un label visible — créant des refuges physiques 
de décompression hors du cadre universitaire, normalisant le fait de 
'ne pas aller bien' dans des espaces neutres."
Pourquoi : Levier culturel pertinent (espaces non-médicaux), ancré dans le quotidien.
Mais reste une adaptation d'un modèle existant. Impact mesuré principalement
sur les personnes qui fréquentent déjà ces espaces — pas de mécanisme
pour toucher les plus réticents à consulter.

IMPACT 85+ — RÉSERVÉ AUX IDÉES VÉRITABLEMENT TRANSFORMATRICES (moins de 3% des soumissions) :
Pour atteindre 85+, une idée DOIT réunir TOUS ces critères simultanément :
✓ Adresse la cause racine du stigma (croyances culturelles, normes familiales, tabou religieux) — pas un symptôme
✓ Touche simultanément plusieurs niveaux : individu + famille + institution
✓ Contient un mécanisme de changement comportemental durable et mesurable
✓ Est adaptée aux spécificités socioculturelles tunisiennes (pas un copié-collé occidental)
✓ Démontre une portée scalable au-delà d'un seul campus ou groupe
Si l'idée ne coche pas les 5 critères, le score DOIT rester sous 85.

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