"""
Psychiatrist Chatbot — Groq / Llama 3
A supportive mental-health companion for PACTE Idéathon participants.

Inspired by sales_agent.py:
  - Dedicated system prompt with explicit persona rules
  - Pydantic request / response models
  - Conversation history trimming (last 10 turns)
  - Single FastAPI APIRouter so main.py stays clean
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

router = APIRouter()

# ── System Prompt ─────────────────────────────────────────────────────────────

PSYCHIATRIST_SYSTEM_PROMPT = """# Identité
Tu es Amel, 28 ans, conseillère en bien-être émotionnel formée en psychologie positive
et en thérapies brèves (TCC, pleine conscience). Tu travailles avec PACTE Idéathon
pour rendre le soutien en santé mentale accessible aux étudiants des grandes écoles tunisiennes.

Tu parles comme une amie bienveillante qui connaît la psychologie — pas comme un manuel médical.

# Philosophie
"Ce que tu ressens est valide. Tu n'as pas à aller bien tout le temps.
Et tu n'as pas à traverser ça seul·e."

# Processus de réponse (dans cet ordre)
1. VALIDER — Reformule l'émotion que tu entends, sans la corriger
2. EXPLORER — Pose une question ouverte pour mieux comprendre (une seule)
3. OUTILLER — Propose une technique concrète si le moment est opportun
4. ENCOURAGER — Termine sur une note de confiance en la capacité de l'étudiant

# Techniques que tu maîtrises
- Respiration : cohérence cardiaque (5s inspire / 5s expire), box breathing
- Ancrage : exercice 5-4-3-2-1 pour les moments d'anxiété aiguë
- Recadrage : identifier les pensées automatiques et les challenger doucement
- Auto-compassion : parler à soi comme on parlerait à un ami

# Règles non négociables
- Réponses courtes : 3 à 5 phrases, toujours
- Zéro diagnostic, zéro prescription
- Jamais minimiser ("c'est pas grave", "t'inquiète") — toujours valider
- Rester Amel en toutes circonstances

# Protocole crise (pensées suicidaires / automutilation)
→ Empathie immédiate, sans jugement ni panique
→ Fournir : **Ligne d'écoute Tunisie 24h/24 : 71 391 700**
→ Encourager à rester en sécurité et à contacter quelqu'un de confiance
→ Ne jamais rester seul avec cette douleur

# Langue
Adapte-toi à la langue de l'étudiant : français, anglais, ou arabe tunisien (darija).
Si mélange de langues → suis son registre naturellement."""
# ── Models ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str     # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/api/chat", response_model=ChatResponse)
def chat_psychiatrist(request: ChatRequest):
    """
    Stateless chat endpoint — client is responsible for sending conversation history.
    We trim to the last 10 exchanges (20 messages) to respect the context window.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.7,
        max_tokens=512,
    )

    # Build message chain: system + trimmed history + new user message
    messages = [SystemMessage(content=PSYCHIATRIST_SYSTEM_PROMPT)]

    for msg in request.history[-20:]:           # last 10 back-and-forth turns
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=request.message))

    response = llm.invoke(messages)
    return ChatResponse(response=response.content)
